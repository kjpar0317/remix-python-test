import httpx

from app.core.config import settings

# 의존성 설정
async def get_openai_client():
    return httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {settings.OPEN_API_KEY}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(30.0) 
    )

async def get_tavily_client():
    return httpx.AsyncClient(
        base_url="https://api.tavily.com",
        headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"},
        timeout=httpx.Timeout(30.0) 
    )

async def get_serp_client():
    return httpx.AsyncClient(
        base_url="https://serpapi.com/search.json",
        timeout=httpx.Timeout(30.0) 
    )
