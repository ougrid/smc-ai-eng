"""Typed application settings.

Every tuning knob is env-configurable (see .env.example at the repo root).
`get_settings()` is an lru_cache'd FastAPI dependency; tests should construct
`Settings(openai_api_key="test", ...)` directly instead of relying on env vars.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is two levels up from this file (backend/app/config.py -> repo root),
# resolved absolutely so it works regardless of the process's working directory.
REPO_ROOT_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT_ENV_FILE, extra="ignore")

    # --- OpenAI (required, no default) ---
    openai_api_key: str
    openai_chat_model: str = "gpt-4o-mini"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dimensions: int = 512

    # --- Postgres ---
    database_url: str = "postgresql+psycopg://app:app_local_dev@localhost:5432/findata"
    agent_ro_database_url: str = (
        "postgresql+psycopg://agent_ro:agent_ro_local_dev@localhost:5432/findata"
    )

    # --- Pinecone local ---
    pinecone_host: str = "http://localhost:5080"
    pinecone_index: str = "10k-filings"

    # --- Auth ---
    jwt_secret: str = "dev-only-change-me"
    jwt_expiry_min: int = 60

    # --- Agent tuning knobs (env-only) ---
    score_floor: float = 0.25
    top_k: int = 10  # bumped from 6 on Day 4: more headroom for boilerplate rejects to still leave real matches
    sql_row_limit: int = 100
    history_max_messages: int = 10  # capped verbatim window fed to route + synthesize -- see agent/history.py

    # --- Reranking (post-Day-5): top_k above is the FINAL evidence count,
    # unchanged in meaning; rerank_pool_size is the wider pre-rerank
    # retrieval breadth (dense + lexical + fusion) the cross-encoder then
    # cuts down to top_k -- see agent/reranked_tool.py ---
    rerank_pool_size: int = 30
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Web ---
    cors_origins: list[str] = ["http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
