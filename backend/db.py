"""SQLAlchemy engine, session, and ORM models."""

from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    JSON, DateTime, ForeignKey, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://optima:optima@localhost:5432/optima"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)


def get_session() -> Session:
    return Session(engine)


# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    jobs: Mapped[list[OptimizationJob]] = relationship(back_populates="user")
    pipelines: Mapped[list[Pipeline]] = relationship(back_populates="user")


# ---------------------------------------------------------------------------
# Optimization jobs
# ---------------------------------------------------------------------------

class OptimizationJob(Base):
    __tablename__ = "optimization_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)   # UUID
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped[User | None] = relationship(back_populates="jobs")


# ---------------------------------------------------------------------------
# Saved pipeline configurations
# ---------------------------------------------------------------------------

class Pipeline(Base):
    __tablename__ = "pipelines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user: Mapped[User] = relationship(back_populates="pipelines")


# ---------------------------------------------------------------------------
# Schema creation helper (called at app startup)
# ---------------------------------------------------------------------------

def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
