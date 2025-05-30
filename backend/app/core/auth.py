import logging

from typing import Optional, MutableMapping, List, Union, Any, Annotated
from datetime import datetime, timedelta, timezone
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import APIKeyCookie
# from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm.session import Session
from jose import jwt, JWTError

from app.models.member import member
from app.core.config import settings
from app.core.security import verify_password

JWTPayloadMapping = MutableMapping[
    str, Union[datetime, bool, str, List[str], List[int]]
]

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")
# 1. 쿠키로 인증하는 Security 정의
cookie_scheme = APIKeyCookie(name="token", auto_error=False)

logger = logging.getLogger(__name__)

async def authenticate(
    *,
    id: str,
    passwd: str,
    db: Session,
) -> Optional[Any]:
    query = member.select().where(member.c.email == id)
    user = await db.fetch_one(query)

    if not user:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "존재하지 않는 유저입니다.")
    if not verify_password(passwd, user.passwd):
        return HTTPException(status.HTTP_403_FORBIDDEN, "인증 정보가 맞지 않습니다.")

    return user


def create_access_token(*, sub: str) -> str:
    return _create_token(
        token_type="access_token",
        lifetime=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        sub=sub,
    )

def _create_token(
    token_type: str,
    lifetime: timedelta,
    sub: str,
) -> str:
    payload = {}
    now = datetime.now(timezone.utc)
    expire = now + lifetime
    # payload["type"] = token_type

    # https://datatracker.ietf.org/doc/html/rfc7519#section-4.1.3
    # The "exp" (expiration time) claim identifies the expiration time on
    # or after which the JWT MUST NOT be accepted for processing
    payload["exp"] = int(expire.timestamp())

    # The "iat" (issued at) claim identifies the time at which the
    # JWT was issued.
    payload["iat"] = int(now.timestamp())

    # The "sub" (subject) claim identifies the principal that is the
    # subject of the JWT
    payload["sub"] = str(sub)

    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.ALGORITHM)

def decode_jwt_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as ex:
        logger.error("JWTError Exception:", repr(ex))  # ❗️여기에서 에러 내용을 정확히 출력
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    except Exception as ex:
        logger.error("Unhandled Exception:", repr(ex))  # ❗️여기에서 에러 내용을 정확히 출력
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during token decoding",
        )

# async def get_token_from_cookie(request: Request, schema_token: Annotated[str | None, Depends(cookie_scheme)]) -> str:
#     # cookie 토큰 우선 (major)
#     token = request.cookies.get("token")

#     logger.info(f"cookie token: {token}")
#     logger.info(f"🟡 schema token: {schema_token}")

#     if token:
#         return token

#     if schema_token:
#         return schema_token

#     # 3. 실패 시 예외
#     raise HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Not authenticated",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
    
async def get_current_user(token: Annotated[str, Depends(cookie_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        logger.info(f"result token: {token}")

        payload = decode_jwt_token(token)
        username = payload.get("sub")

        if username is None:
            raise credentials_exception
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token sub")
    return token