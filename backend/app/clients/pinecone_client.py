"""Pinecone-local client helpers.

Host resolution must never hardcode a port -- pinecone-local assigns each
index a data-plane port in 5081-5090 at creation time. This mirrors the
resolution logic in scripts/load_pinecone.py; keep the two in sync.
"""

from typing import Any

from pinecone import Pinecone

from app.config import Settings


def resolve_index(settings: Settings) -> Any:
    """Connect to the control plane and return a data-plane Index handle."""
    pc = Pinecone(api_key="pclocal", host=settings.pinecone_host)
    host = pc.describe_index(settings.pinecone_index).host
    if not host.startswith("http"):
        host = f"http://{host}"
    return pc.Index(host=host)


def vector_count(index: Any) -> int:
    return index.describe_index_stats().total_vector_count
