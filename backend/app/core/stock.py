import pandas as pd
import yfinance as yf
import logging
from typing import List, Dict, Any
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

def get_currency_rate(base_currency: str, target_currency: str) -> float:
    if base_currency == target_currency:
        return 1.0
    
    # 환율 데이터 가져오기 (예: USD/KRW의 경우 'USDKRW=X')
    currency_pair = f"{base_currency}{target_currency}=X"
   # logger.info(f"currency_pair: {currency_pair}")

    try:
        currency_data = yf.download(currency_pair, period='1d')['Close'][currency_pair]
        # logger.info(f"currency_data: {currency_data}")
        # logger.info(f"currency_data rate: {currency_data.iloc[-1]}")

        return currency_data.iloc[-1] if not currency_data.empty else 1.0
    except Exception:
        # logger.info("no currency_data")
        return 1.0
    
def subtract_timeframe(end_date: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    number = int(''.join(filter(str.isdigit, timeframe)))
    
    if 'mo' in timeframe:
        return end_date - relativedelta(months=number)
    elif 'y' in timeframe:
        return end_date - relativedelta(years=number)
    elif 'd' in timeframe:
        return end_date - pd.Timedelta(days=number)
    else:
        raise ValueError(f"Unsupported timeframe unit in {timeframe}")

def get_final_rsi_recommendation(recommendations):
    strong_buy = recommendations.count("강력매수")
    buy = recommendations.count("매수")
    # neutral = recommendations.count("중립")
    sell = recommendations.count("매도")
    strong_sell = recommendations.count("강력매도")
    
    # 가중치 부여
    buy_score = (strong_buy * 2) + buy
    sell_score = (strong_sell * 2) + sell
    
    if buy_score > sell_score and strong_buy > 0:
        return "강력매수"
    elif buy_score > sell_score:
        return "매수"
    elif sell_score > buy_score and strong_sell > 0:
        return "강력매도"
    elif sell_score > buy_score:
        return "매도"
    else:
        return "중립"
    
def generate_recommendations(content: str) -> List[Dict[str, Any]]:
    # 투자 추천 생성
    short_term_confidence = 0.65 if "긍정적" in content else 0.45
    long_term_confidence = 0.85 if "성장" in content else 0.55
    
    return [
        {
            "type": "short_term",
            "action": "매수" if short_term_confidence > 0.5 else "관망",
            "confidence": short_term_confidence
        },
        {
            "type": "long_term",
            "action": "매수" if long_term_confidence > 0.5 else "관망",
            "confidence": long_term_confidence
        }
    ]

def analyze_news_sentiment(content: str) -> Dict[str, Any]:
    # 뉴스 감성 분석
    positive_words = ["상회", "성장", "긍정적", "상승"]
    negative_words = ["손실", "하락", "부정적", "음수"]
    
    positive_count = sum(1 for word in positive_words if word in content)
    negative_count = sum(1 for word in negative_words if word in content)
    total = positive_count + negative_count
    
    if total == 0:
        sentiment_score = 0.5
    else:
        sentiment_score = positive_count / total
    
    # 감성 점수에 따른 설명 텍스트 생성
    sentiment_description = {
        (0.8, 1.0): "매우 긍정적",
        (0.6, 0.8): "긍정적",
        (0.4, 0.6): "중립적",
        (0.2, 0.4): "부정적",
        (0.0, 0.2): "매우 부정적"
    }
    
    description = next(
        text for (lower, upper), text in sentiment_description.items()
        if lower <= sentiment_score <= upper
    )
    
    key_factors = extract_key_factors(content)
    
    return {
        "overall_score": round(sentiment_score, 2),
        "sentiment_text": description,
        "recent_trend": "상승" if sentiment_score > 0.6 else "하락" if sentiment_score < 0.4 else "중립적",
        "key_factors": key_factors
    }

def extract_key_factors(content: str) -> list:
    """
    텍스트에서 주요 요인을 추출하는 함수
    """
    key_factors = []
    
    # 실적 관련 요인
    if any(word in content for word in ["실적", "매출", "이익", "순이익"]):
        if "상회" in content or "증가" in content:
            key_factors.append("실적 개선")
        elif "하회" in content or "감소" in content:
            key_factors.append("실적 부진")
    
    # 현금 흐름 관련
    if "현금" in content:
        if "증가" in content or "양호" in content:
            key_factors.append("현금흐름 양호")
        elif "감소" in content or "부족" in content:
            key_factors.append("현금흐름 악화")
    
    # 시장 전망 관련
    if "전망" in content or "예상" in content:
        if "긍정" in content or "상승" in content:
            key_factors.append("긍정적 시장 전망")
        elif "부정" in content or "하락" in content:
            key_factors.append("부정적 시장 전망")
    
    # 애널리스트 의견 관련
    if "애널리스트" in content:
        if "매수" in content or "긍정" in content:
            key_factors.append("긍정적 애널리스트 평가")
        elif "매도" in content or "부정" in content:
            key_factors.append("부정적 애널리스트 평가")
    
    # 기본값 설정
    if not key_factors:
        key_factors = ["충분한 요인 식별 불가"]
    
    return key_factors[:4] 