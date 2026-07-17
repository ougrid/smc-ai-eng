"""The users table -- the only table the LLM-generated-SQL path (agent_ro
role) is deliberately unable to read; see scripts/initdb/02_roles.sql."""

import uuid
from datetime import datetime

from sqlalchemy import String, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(50), server_default="user")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
