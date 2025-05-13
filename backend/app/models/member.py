from sqlalchemy import Table, Column, String
from app.core.database import metadata

member = Table(
    "member",
    metadata,
    Column("email", String(50), primary_key=True),
    Column("passwd", String(200))
)