import os
import pathlib

from pydantic import AnyHttpUrl, EmailStr, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Union
from dotenv import load_dotenv

# env 파일을 읽어들이자.
load_dotenv()

# Project Directories
ROOT = pathlib.Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    JWT_SECRET: str = "TEST_SECRET_DO_NOT_USE_IN_PROD_PYTHON_BACKEND-TEST_SECRET_DO_NOT_USE_IN_PROD_PYTHON_BACKEND"
    ALGORITHM: str = "HS256"

    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins COR URL 입력한다.
    # e.g: '["http://localhost", "http://localhost:4200", "http://localhost:3000", \
    # "http://localhost:8080", "http://local.dockertoolbox.tiangolo.com"]'
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    SQLALCHEMY_DATABASE_URI: Optional[str] = "sqlite:///stock.db"
    FIRST_SUPERUSER: EmailStr = "test@test.com"
    FIRST_SUPERUSER_PW: str = "test"
    OPEN_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SERP_API_KEY: str = os.getenv("SERP_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")

    class Config:
        case_sensitive = True


settings = Settings()