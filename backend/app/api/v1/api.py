from fastapi import APIRouter

from app.api.v1.endpoints import user, stock

api_router = APIRouter()
# api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(user.router, prefix="/user", tags=["user"])
api_router.include_router(stock.router, prefix="/stock", tags=["stock"])
