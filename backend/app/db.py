"""Engine builders.

Two distinct engines exist on purpose: the app-role engine (used by auth /
general app code) and the agent_ro-role engine (the ONLY engine the
LLM-generated-SQL path may ever receive). Never conflate the two -- the agent
must physically be unable to reach the `users` table, which is enforced at
the Postgres role level (see scripts/initdb/02_roles.sql), not just in code.
"""

from typing import Iterator

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session


class Base(DeclarativeBase):
    pass


def build_engine(url: str) -> Engine:
    """App-role engine (read/write, e.g. for the users table)."""
    return create_engine(url, pool_pre_ping=True)


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency yielding a session against the app-role engine."""
    with Session(request.app.state.engine) as session:
        yield session


def build_agent_ro_engine(url: str) -> Engine:
    """agent_ro-role engine -- read-only at both the SQL-role and session level.

    `default_transaction_read_only=on` is already set on the role itself
    (02_roles.sql), but pinning it again here via connect_args is cheap
    defense-in-depth against a future role/grant change.
    """
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"options": "-c default_transaction_read_only=on"},
    )
