"""
app/db/models.py
----------------
SQLAlchemy ORM models for task history persistence.

Schema is created on startup via init_db() — no Alembic for the MVP.
Bump table-name or add a migration tool when the schema starts to evolve.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class TaskRecord(Base):
    """One row per detection task. Mirrors TaskState plus timestamps."""

    __tablename__ = "detection_tasks"

    task_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    video_id: Mapped[str] = mapped_column(String(36), nullable=False)
    video_filename: Mapped[Optional[str]] = mapped_column(String(512))
    prompt: Mapped[str] = mapped_column(Text, nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[Optional[str]] = mapped_column(Text)
    zip_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    early_terminated: Mapped[bool] = mapped_column(Boolean, default=False)
    termination_reason: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
