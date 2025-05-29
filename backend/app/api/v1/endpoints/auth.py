from fastapi import APIRouter, HTTPException, Form

from app.core.auth import authenticate, create_access_token
from app.core.database import database
from app.schemas.auth import AuthRS

router = APIRouter()

# 엔드포인트 정의
@router.post("/login", response_model=AuthRS)
async def login(username: str = Form(...), password: str = Form(...)):
    member = await authenticate(id=username, passwd=password, db=database)

    if member is None:
        raise HTTPException(404, "로그인 실패")
    
    return { "token_type": "bearer", "access_token": create_access_token(sub=member.email) }