from fastapi import FastAPI
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import database
from app.api.v1.api import api_router

# 미리 env 파일을 읽어들이자.
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.connect()
    yield
    await database.disconnect()

app = FastAPI(title="주식 AI 분석 플랫폼", lifespan=lifespan)

app.include_router(api_router, prefix=settings.API_V1_STR)
# app.include_router(root_router)