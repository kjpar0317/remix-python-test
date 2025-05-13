import httpx
import os

# 의존성 설정
async def get_openai_client():
    return httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"},
        timeout=httpx.Timeout(30.0) 
    )

async def get_tavily_client():
    return httpx.AsyncClient(
        base_url="https://api.tavily.com",
        headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}"},
        timeout=httpx.Timeout(30.0) 
    )
