from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import database
from app.api.v1.api import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(title="주식 AI 분석 플랫폼", lifespan=lifespan)

app.include_router(api_router, prefix=settings.API_V1_STR)
# app.include_router(root_router)