from pydantic import BaseModel

# 모델 정의
class AuthRS(BaseModel):
    accessToken: str
