import httpx
from fastapi import APIRouter, Depends, HTTPException
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.tracers import LangChainTracer
import httpx
import yfinance as yf
from app.schemas.stock import StockAnalysisResponse, StockAnalysisRequest
from app.core.http import get_openai_client, get_tavily_client
from app.core.stock import generate_recommendations, analyze_news_sentiment

router = APIRouter()
llm = ChatOpenAI()

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
