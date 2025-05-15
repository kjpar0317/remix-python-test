from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import authenticate, create_access_token
from app.core.database import database
from app.schemas.member import Member
from app.schemas.auth import AuthRS

router = APIRouter()

# 엔드포인트 정의
@router.post("/login", response_model=AuthRS)
async def login(request: Member):
    print(request)

    member = await authenticate(id=request.email, passwd=request.passwd, db=database)

    if member is None:
        raise HTTPException(404, "로그인 실패")
    
    return { "accessToken": create_access_token(sub=member.email) }