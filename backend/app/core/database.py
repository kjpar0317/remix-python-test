from sqlalchemy.ext.asyncio import create_async_engine # , AsyncSession
# from sqlalchemy.orm import sessionmaker
from sqlalchemy import MetaData
from databases import Database
from app.core.config import settings

# 비동기 데이터베이스 연결 설정
database = Database(settings.SQLALCHEMY_DATABASE_URI)
metadata = MetaData()

# print(settings.DATABASE_URL)

# # 비동기 엔진 생성
# engine = create_async_engine(settings.DATABASE_URL, echo=True)

# async_session = sessionmaker(
#     engine, expire_on_commit=False, class_=AsyncSession
# )