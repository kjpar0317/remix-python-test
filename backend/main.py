from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import database
from app.core.auth import decode_jwt_token
from app.api.v1.api import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(title="AI 분석 플랫폼", lifespan=lifespan)

# ✅ 전역 의존성으로 JWT 적용
@app.middleware("http")
async def jwt_global_middleware(request: Request, call_next):
    # Swagger, 로그인, 루트 등 제외하고 토큰 검사
    excluded_paths = ["/docs", "/openapi.json", "/redoc", "/api/v1/auth/login", "/"]
    if any(request.url.path.startswith(path) for path in excluded_paths):
        return await call_next(request)

    auth = request.headers.get("Authorization")

    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth.split(" ")[1]
    try:
        decode_jwt_token(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

    return await call_next(request)

app.include_router(api_router, prefix=settings.API_V1_STR)
# app.include_router(root_router)