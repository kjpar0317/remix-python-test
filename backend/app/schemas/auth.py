from pydantic import BaseModel

# 모델 정의
class AuthRS(BaseModel):
    token_type: str
    access_token: str
