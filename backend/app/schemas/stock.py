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
