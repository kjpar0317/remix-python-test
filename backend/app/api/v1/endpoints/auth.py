from fastapi import APIRouter, Response, HTTPException, Form, status

from app.core.auth import authenticate, create_access_token
from app.core.database import database
from app.schemas.auth import AuthRS

router = APIRouter()

# 엔드포인트 정의
@router.post("/login", response_model=AuthRS)
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    member = await authenticate(id=username, passwd=password, db=database)
    
    token = create_access_token(sub=member.email)

    response.set_cookie(
        key="token",
        value=token,  # 실제로는 JWT 발급
        httponly=True,
        secure=False,         # HTTPS 환경에서만 작동 (로컬 테스트 시 False로)
        samesite="Lax"       # 또는 "None" (크로스 도메인 지원 시 필요)
    )

    return { "token_type": "bearer", "access_token": token }

@router.post("/logout")
async def logout(response: Response) -> Response:
    response.delete_cookie(
        key="token",
        httponly=True,
        secure=False,  # 로그인 때 설정한 값과 동일하게 맞춰야 함
        samesite="Lax"
    )
    
    return {"message": "Logged out"}
