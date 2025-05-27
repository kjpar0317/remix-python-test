from sqlalchemy import MetaData
from databases import Database
from app.core.config import settings

# 비동기 데이터베이스 연결 설정
database = Database(settings.SQLALCHEMY_DATABASE_URI)
metadata = MetaData()