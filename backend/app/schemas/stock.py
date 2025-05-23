from pydantic import BaseModel
from typing import List, Dict, Any
# 모델 정의
class StockAnalysisRequest(BaseModel):
    ticker: str
    timeframe: str = "1y"
    analysis_type: List[str] = ["fundamental", "technical", "sentiment"]

class StockAnalysisResponse(BaseModel):
    ticker: str
    analysis: str
    recommendations: List[Dict[str, Any]]
    news_sentiment: Dict[str, Any]


class StockRequest(BaseModel):
    ticker: str
    timeframe: str = "6mo"  # e.g., "1mo", "6mo", "1y"
    currency: str = "USD"

class TunningPoint(BaseModel):
    date: str
    type: str
    price: float
    
# 주식 예측 결과를 반환할 Pydantic 모델
class PredictionData(BaseModel):
    dates: List[str]
    close: List[float]
    lstmClose: List[float]
    cnnClose: List[float]
    ma200: List[float]
    goldenCross: List[int]
    rsi: List[float]
    upperBand: List[float]
    lowerBand: List[float]
    bollingerBreakoutUpper: List[float]
    bollingerBreakoutLower: List[float]
    macd: List[float]
    signal: List[float]
    sniperSignal: List[float]
    smartSniper: List[float]
    # macdSignal: List[str]
    doubleBottom: List[float]
    doubleTop: List[float]
    headAndShoulders: List[float]
    inverseHeadAndShoulders: List[float]
    tunningPoints: List[TunningPoint]
    recommendMacdSignal: str
    recommendGC: str
    recommendRSI: str
    recommendUpperLower: str
    recommendSniperSignal: str
    recommendTotalDecision: str