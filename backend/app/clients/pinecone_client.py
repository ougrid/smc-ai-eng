"""Pinecone-local client helpers.

Host resolution must never hardcode a port -- pinecone-local assigns each
index a data-plane port in 5081-5090 at creation time. This mirrors the
resolution logic in scripts/load_pinecone.py; keep the two in sync.
"""

from typing import Any

from pinecone import Pinecone

from app.config import Settings


def resolve_index(settings: Settings) -> Any:
    """Connect to the control plane and return a data-plane Index handle.

    describe_index() reports the host as `https://...`, but pinecone-local's
    data plane does not actually speak TLS -- force http regardless of
    whatever scheme (or lack of one) is reported. Keep in sync with
    scripts/load_pinecone.py's resolve_data_plane_host.
    """
    pc = Pinecone(api_key="pclocal", host=settings.pinecone_host)
    host = pc.describe_index(settings.pinecone_index).host
    host = host.split("://", 1)[-1]
    return pc.Index(host=f"http://{host}")


def vector_count(index: Any) -> int:
    return index.describe_index_stats().total_vector_count
