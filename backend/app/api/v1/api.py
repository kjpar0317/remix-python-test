from fastapi import APIRouter, Depends

from app.api.v1.endpoints import user, auth, stock
from app.core.auth import get_current_user

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(user.router, prefix="/user", tags=["user"], dependencies=[Depends(get_current_user)])
api_router.include_router(stock.router, prefix="/stock", tags=["stock"], dependencies=[Depends(get_current_user)])
