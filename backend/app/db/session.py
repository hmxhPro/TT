"""
app/db/session.py
-----------------
Async SQLAlchemy engine + session factory for PostgreSQL.

The engine is created at import time using settings.DATABASE_URL. Connection
is lazy (asyncpg pool is established on first use), so a misconfigured or
unreachable DB does NOT crash module import — failures surface at the first
query and are swallowed by the persistence layer's try/except.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import logger

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def init_db() -> None:
    """Create tables if missing. MVP: no Alembic migration tracking."""
    from app.db.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("DB schema ensured (detection_tasks)")
