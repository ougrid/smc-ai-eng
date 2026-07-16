from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.clients.pinecone_client import resolve_index, vector_count
from app.config import Settings, get_settings
from app.db import build_agent_ro_engine, build_engine

EXPECTED_POSTGRES_ROWS = 192
EXPECTED_VECTOR_COUNT = 4072


def create_app(
    *,
    settings: Settings | None = None,
    graph=None,
    engine=None,
    agent_engine=None,
    pinecone_index=None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.graph = graph
        app.state.engine = engine if engine is not None else build_engine(settings.database_url)
        app.state.agent_engine = (
            agent_engine
            if agent_engine is not None
            else build_agent_ro_engine(settings.agent_ro_database_url)
        )
        app.state.pinecone_index = (
            pinecone_index if pinecone_index is not None else resolve_index(settings)
        )
        yield

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_headers=["Authorization", "Content-Type"],
        allow_methods=["*"],
        expose_headers=["x-vercel-ai-ui-message-stream"],
    )

    @app.get("/api/health")
    def health():
        try:
            with app.state.engine.connect() as conn:
                postgres_rows = conn.execute(text("SELECT COUNT(*) FROM financial_data")).scalar_one()
        except Exception:
            postgres_rows = 0

        try:
            vc = vector_count(app.state.pinecone_index)
        except Exception:
            vc = 0

        ok = postgres_rows == EXPECTED_POSTGRES_ROWS and vc == EXPECTED_VECTOR_COUNT
        return {"postgres_rows": postgres_rows, "vector_count": vc, "ok": ok}

    return app
