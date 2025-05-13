from fastapi import APIRouter
from app.models.member import member
from app.core.database import database

router = APIRouter()

@router.get("")
async def users():
    query = member.select()
    rows = await database.fetch_all(query)

    return rows