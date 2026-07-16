from pinecone import Pinecone


def resolve_index_host(pc: Pinecone, index_name: str) -> str:
    """Resolve the data-plane host for an index via describe_index.

    pinecone-local assigns each index its own data-plane port (5081-5090);
    it must never be hardcoded — always read back from describe_index().host.
    """
    host = pc.describe_index(index_name).host
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host


def resolve_index(settings) -> "Pinecone.Index":
    """Build a Pinecone client against pinecone-local and return a ready-to-query index handle."""
    pc = Pinecone(api_key="pclocal", host=settings.pinecone_host)
    host = resolve_index_host(pc, settings.pinecone_index)
    return pc.Index(host=host)


def vector_count(index) -> int:
    return index.describe_index_stats().total_vector_count
