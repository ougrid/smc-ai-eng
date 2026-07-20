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

from app.agent.coverage import build_coverage
from app.agent.graph import build_graph
from app.agent.nodes.route import build_route_llm
from app.agent.nodes.sql_retrieve import build_sql_llm
from app.agent.nodes.synthesize import build_synthesis_llm
from app.agent.sql_tool import SqlTool
from app.agent.vector_tool import VectorTool
from app.auth.models import User  # noqa: F401 -- import registers the table with Base.metadata
from app.auth.router import router as auth_router
from app.chat.router import router as chat_router
from app.clients.llm_client import build_embedder, build_llm
from app.clients.pinecone_client import resolve_index, vector_count
from app.config import Settings, get_settings
from app.db import Base, build_agent_ro_engine, build_engine


def _build_real_graph(settings: Settings, agent_engine: Engine, pinecone_index: Any):
    """Wires the real LangGraph agent -- only called when no `graph` stub
    was injected, so offline tests never construct a live LLM/embedder."""
    llm = build_llm(settings)
    embedder = build_embedder(settings)
    return build_graph(
        coverage=build_coverage(agent_engine),
        route_llm=build_route_llm(llm),
        sql_llm=build_sql_llm(llm),
        synth_llm=build_synthesis_llm(llm),
        sql_tool=SqlTool(agent_engine, row_limit=settings.sql_row_limit),
        vector_tool=VectorTool(
            pinecone_index, embedder, top_k=settings.top_k, score_floor=settings.score_floor
        ),
        history_max_messages=settings.history_max_messages,
    )


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
        Base.metadata.create_all(app.state.engine)
        app.state.agent_engine = (
            agent_engine
            if agent_engine is not None
            else build_agent_ro_engine(resolved_settings.agent_ro_database_url)
        )
        app.state.pinecone_index = (
            pinecone_index if pinecone_index is not None else resolve_index(resolved_settings)
        )
        app.state.graph = graph if graph is not None else _build_real_graph(
            resolved_settings, app.state.agent_engine, app.state.pinecone_index
        )
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

    app.include_router(auth_router)
    app.include_router(chat_router)

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
