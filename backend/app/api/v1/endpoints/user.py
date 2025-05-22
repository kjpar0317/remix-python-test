from fastapi import APIRouter, Depends
from app.models.member import member
from app.core.database import database
from app.core.auth import cookie_scheme, get_current_user

router = APIRouter()

@router.get("")
async def users():
    query = member.select()
    rows = await database.fetch_all(query)

    return rows

@router.get("/me", dependencies=[Depends(cookie_scheme)])
async def get_me(token: str = Depends(get_current_user)):
    return {"token": token}