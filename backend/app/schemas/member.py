from pydantic import BaseModel

# 모델 정의
class Member(BaseModel):
    email: str
    passwd: str