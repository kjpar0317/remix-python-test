import httpx
from fastapi import APIRouter, Depends, HTTPException
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.tracers import LangChainTracer
from datetime import timedelta
import yfinance as yf
import pandas as pd

from app.schemas.stock import StockAnalysisResponse, StockAnalysisRequest, StockRequest, PredictionData
from app.core.http import get_openai_client, get_tavily_client
from app.core.stock import subtract_timeframe, get_final_rsi_recommendation, generate_recommendations, analyze_news_sentiment
from app.core.stock_indicators import calc_price_with_ta, predict_close_price_with_rf, calc_price_with_lstm_cnn

router = APIRouter()

# 엔드포인트 정의
@router.post("/analysis", response_model=StockAnalysisResponse)
async def analyze_stock(
    request: StockAnalysisRequest, 
    openai_client: httpx.AsyncClient = Depends(get_openai_client),
    tavily_client: httpx.AsyncClient = Depends(get_tavily_client),
):
    # 1. Tavily API로 주식 관련 정보 수집
    news_response = await tavily_client.post(
        "/search",
        json={
            "query": f"{request.ticker} stock financial news analysis",
            "search_depth": "advanced",
            "include_domains": ["finance.yahoo.com", "seekingalpha.com", "marketwatch.com", "bloomberg.com"]
        }
    )

    if news_response.status_code != 200:
        raise HTTPException(status_code=503, detail="뉴스 데이터 검색에 실패했습니다")

    news_data = news_response.json()
    
    # 2. OpenAI API로 수집된 정보 분석
    analysis_response = await openai_client.post(
        "/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a financial analyst expert."},
                {"role": "user", "content": f"Analyze this stock data for {request.ticker} and provide insights: {news_data}"}
            ],
            "temperature": 0.2
        }
    )
 
    if analysis_response.status_code != 200:
        raise HTTPException(status_code=503, detail="데이터 분석에 실패했습니다")
    
    analysis_result = analysis_response.json()
    analysis_content = analysis_result["choices"][0]["message"]["content"]

    tracer = LangChainTracer(
        project_name="stock_analysis"
    )

    # ChatOpenAI 모델 설정
    chat = ChatOpenAI(
        # api_key=settings.OPEN_API_KEY,
        temperature=0,
        callbacks=[tracer],
        metadata = { "analysis_type": request.analysis_type }
    )

    financeTicker = yf.Ticker(request.ticker)

    # 입력 메시지 구성
    messages = [
        SystemMessage(
            content="한글로 설명해줘"
        ),
        HumanMessage(
            content=f"Analyze the following stock data for {request.ticker}: {analysis_content}"
        ),
        HumanMessage(
            content=f"yahoo - financials: {financeTicker.financials}, balance_sheet: {financeTicker.balance_sheet}"
        ),
        HumanMessage(
            content=f"yahoo - history: {financeTicker.history(period=request.timeframe)[['Close', 'Volume']]}"
        ),
        HumanMessage(
            content=f"위에 데이터를 통해 이 주식의 {",".join(request.analysis_type)} analysis도 포함해서 설명해줘"
        ),        
    ]

    # 분석 실행
    langsmith_result = await chat.ainvoke(messages)
    
    content = langsmith_result.content
    
    # 텍스트에서 주요 정보 추출
    structured_response = {
        "ticker": request.ticker,
        "analysis": content,
        "recommendations": generate_recommendations(content),
        "news_sentiment": analyze_news_sentiment(content)
    }

    return structured_response


"""
Golden Cross (recommendGC: 매도):

Golden Cross가 매도를 추천하고 있다는 것은, 이동평균선 골든크로스가 발생하지 않았거나 발생해도 가격이 상승하지 않는 상태일 수 있습니다. 즉, 상승 추세가 아닌 하락 추세가 계속될 가능성을 의미합니다.

RSI (recommendRSI: 강력매수):

RSI가 강력매수로 추천된 것은, 과매도 상태(RSI < 30)가 나타났다는 것입니다. 즉, 현재 주식이 너무 저평가되어 있으며, 가격 반등의 가능성이 있는 상태라는 신호입니다.

Upper/Lower Band (recommendUpperLower: 매도):

Upper Band와 Lower Band를 기준으로 매도를 추천하고 있다는 것은, 가격이 상단 밴드를 넘어섰거나 하단 밴드에 도달한 상태로 볼 수 있습니다. 가격이 너무 높거나, 너무 낮은 위치에 있어 추가적인 상승이나 하락을 피하고 매도할 시점이라는 신호일 수 있습니다.

Sniper Signal (recommendSniperSignal: 매도):

Sniper Signal이 매도를 추천하고 있다는 것은, 특정한 매도 신호(예: 고급 기술적 분석 신호)가 발생했다는 것을 의미합니다. 매도 타이밍을 나타내는 강한 신호가 주어진 상태입니다.

종합적으로 볼 때:
RSI가 강력매수를 추천하는 것과는 달리, Golden Cross, Upper/Lower Band, Sniper Signal이 모두 매도를 추천하고 있습니다. 이는 과매도 상태에서의 반등 가능성을 나타내는 RSI 신호와 하락 추세를 나타내는 다른 지표들 사이에서 상반된 신호가 발생하는 상황입니다.

이런 경우는 리스크가 큰 상태로 해석할 수 있으며, 매수를 고려하기엔 위험할 수 있습니다. 매도 신호가 더 우세한 상황으로 보이므로, 전체적인 흐름은 하락 추세에 가까운 것으로 해석됩니다.

결론:
현재는 하락 추세가 우세한 시점으로 보이며, 매도가 더 적합한 전략일 수 있습니다. 다만, RSI의 강력매수 신호를 고려하여 단기적인 반등을 기대할 수는 있지만, 전체적인 기술적 지표들은 하락을 시사하고 있으므로 신중한 판단이 필요합니다.
"""
@router.post("/chart-data", response_model=PredictionData, response_model_by_alias=True)
async def chart_data(req: StockRequest):
    ticker = req.ticker
    stock_data = yf.Ticker(ticker)

    end_date = pd.Timestamp.today()
    real_start_date = subtract_timeframe(end_date, req.timeframe)
    start_date = real_start_date - timedelta(days=200)
    # df = stock_data.history(start=start_date, end=end_date)
    df = stock_data.history(start=real_start_date, end=end_date)

    df['Date'] = df.index.strftime('%Y-%m-%d').tolist()
    # df["LSTM Close"] = df["Close"]
    # df["CNN Close"] = df["Close"]

    # ta 계산
    df = calc_price_with_ta(df)

    # ta 계산 후 200일치 + 된 거 날림
    real_start_date = real_start_date.tz_localize(df.index.tz)
    df = df[df.index >= real_start_date]

    # df = df.where(pd.notnull(df), 0)
    # NaN → 0 처리
    df.fillna(0, inplace=True)

    # 미래 예측치 더함
    df = predict_close_price_with_rf(df)

    print(df)

    # 각 영역에 대한 추천 값을 저장
    recommend_gc = []
    recommend_rsi = []
    recommend_upper_lower = []  # Upper Band와 Lower Band를 합친 추천 리스트
    recommend_sniper_signal = []  # Sniper Signal에 대한 추천

    # 데이터프레임에서 각 값 계산
    for index, row in df.iterrows():
        latest_rsi = row['RSI']
        latest_gc = row['Golden Cross']
        latest_close = row['Close']
        latest_upper = row['Upper Band']
        latest_lower = row['Lower Band']
        latest_sniper_signal = row['Sniper Signal']

        # Golden Cross에 대한 추천
        if latest_gc == 1:
            recommend_gc.append("매수")
        else:
            recommend_gc.append("매도")

        # RSI에 대한 추천
        if latest_rsi < 30:
            recommend_rsi.append("강력매수")
        elif latest_rsi < 40:
            recommend_rsi.append("매수")
        elif latest_rsi > 60:
            recommend_rsi.append("매도")            
        elif latest_rsi > 70:
            recommend_rsi.append("강력매도")

        # Upper Band와 Lower Band에 대한 추천
        if latest_close > latest_upper:
            recommend_upper_lower.append("매도")
        else:
            recommend_upper_lower.append("매수")

        if latest_close < latest_lower:
            recommend_upper_lower.append("매수")
        else:
            recommend_upper_lower.append("매도")

        # Sniper Signal에 대한 추천
        if latest_sniper_signal == 1:  # Sniper Signal이 1이면 매수
            recommend_sniper_signal.append("매수")
        else:  # Sniper Signal이 0이면 매도
            recommend_sniper_signal.append("매도")

    # 1년치 데이터에 대해 각 추천을 종합하여 최종 추천을 내림
    final_recommend_gc = "매수" if recommend_gc.count("매수") > recommend_gc.count("매도") else "매도"
    final_recommend_rsi = get_final_rsi_recommendation(recommend_rsi)
    final_recommend_upper_lower = "매수" if recommend_upper_lower.count("매수") > recommend_upper_lower.count("매도") else "매도"
    final_recommend_sniper_signal = "매수" if recommend_sniper_signal.count("매수") > recommend_sniper_signal.count("매도") else "매도"

    # print(df)

    # 예측 데이터 반환
    result = {
        "dates": df['Date'].tolist(),
        "close": df['Close'].tolist(),
        "lstmClose": df['LSTM Close'].tolist(),
        "cnnClose": df['CNN Close'].tolist(),
        "ma200": df['MA200'].tolist(),
        "goldenCross": df['Golden Cross'].tolist(),
        "rsi": df['RSI'].tolist(),
        "upperBand": df['Upper Band'].tolist(),
        "lowerBand": df['Lower Band'].tolist(),
        "bollingerBreakoutUpper": df["Bollinger Breakout Upper"].tolist(),
        "bollingerBreakoutLower": df["Bollinger Breakout Lower"].tolist(),
        "sniperSignal": df['Sniper Signal'].tolist(),
        "smartSniper": df['Smart Sniper'].tolist(),
        "doubleBottom": df["Double Bottom"].tolist(),
        "doubleTop": df["Double Top"].tolist(),
        "headAndShoulders": df["Head and Shoulders"].tolist(),
        "inverseHeadAndShoulders": df["Inverse Head and Shoulders"].tolist(),
        "recommendGC": final_recommend_gc,  # Golden Cross에 대한 최종 추천
        "recommendRSI": final_recommend_rsi,  # RSI에 대한 최종 추천
        "recommendUpperLower": final_recommend_upper_lower,  # Upper Band에 대한 최종 추천
        "recommendSniperSignal": final_recommend_sniper_signal  # Lower Band에 대한 최종 추천
    }

    return result

@router.post("/test")
async def post_test(req: StockRequest):
    ticker = req.ticker
    stock_data = yf.Ticker(ticker)
    end_date = pd.Timestamp.today()
    real_start_date = subtract_timeframe(end_date, req.timeframe)
    # start_date = real_start_date - timedelta(days=200)

    # print(f"start_date: {start_date}")

    df = stock_data.history(start=real_start_date, end=end_date)
    # df = stock_data.history(start=real_start_date, end=end_date)

    print(df)

    print("-----------------------------------------")

    real_start_date = real_start_date.tz_localize(df.index.tz)
    df = df[df.index >= real_start_date]

    print(df)

    df['Date'] = df.index.strftime('%Y-%m-%d').tolist()




    # print(calc_price_with_lstm_cnn(df))

    return True