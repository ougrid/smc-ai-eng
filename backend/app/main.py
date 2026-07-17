"""FastAPI app factory.

`create_app()` builds real clients (Postgres engines, pinecone-local index)
only inside `lifespan`, and only for arguments left as `None` -- callers
(tests) can inject stubs for any of them and the app never touches the
network or a real database. This is what keeps the test suite offline and
free to run.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine, text

from app.clients.pinecone_client import resolve_index, vector_count
from app.config import Settings, get_settings
from app.db import build_agent_ro_engine, build_engine


def create_app(
    *,
    settings: Settings | None = None,
    engine: Engine | None = None,
    agent_engine: Engine | None = None,
    pinecone_index: Any | None = None,
    graph: Any | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = resolved_settings
        app.state.engine = (
            engine if engine is not None else build_engine(resolved_settings.database_url)
        )
        app.state.agent_engine = (
            agent_engine
            if agent_engine is not None
            else build_agent_ro_engine(resolved_settings.agent_ro_database_url)
        )
        app.state.pinecone_index = (
            pinecone_index if pinecone_index is not None else resolve_index(resolved_settings)
        )
        app.state.graph = graph
        yield

    app = FastAPI(title="smc-ai-eng", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["x-vercel-ai-ui-message-stream"],
    )

    @app.get("/api/health")
    def health() -> dict:
        with app.state.engine.connect() as conn:
            postgres_rows = conn.execute(
                text("SELECT COUNT(*) FROM financial_data")
            ).scalar_one()
        vc = vector_count(app.state.pinecone_index)
        return {
            "postgres_rows": postgres_rows,
            "vector_count": vc,
            "ok": postgres_rows == 192 and vc == 4072,
        }

    return app


app = create_app()
