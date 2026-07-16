"""Idempotent loader: streams data/pinecone_vectors.jsonl.gz into pinecone-local.

Safe to re-run after every `docker compose up` (pinecone-local has no persistence
across restarts): creates the index if missing, then upserts by the vectors'
existing stable ids.
"""

import gzip
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from pinecone import Pinecone, ServerlessSpec  # noqa: E402

from app.clients.pinecone_client import resolve_index_host  # noqa: E402
from app.config import get_settings  # noqa: E402

VECTORS_PATH = REPO_ROOT / "data" / "pinecone_vectors.jsonl.gz"
BATCH_SIZE = 200
EXPECTED_COUNT = 4072

# Canonical company key mapping — deliberately matches the SQL table's company
# names ("Google", not "Alphabet"), not the PDF filenames.
COMPANY_BY_FILE = {
    "Alphabet_10K_FY2025.pdf": "Google",
    "Amazon_10K_FY2025.pdf": "Amazon",
    "Apple_10K_FY2025.pdf": "Apple",
    "Meta_10K_FY2025.pdf": "Meta",
}

JUNK_METADATA_KEYS = {"creator", "producer", "moddate", "creationdate"}


def wait_until_ready(pc: Pinecone, retries: int = 30, delay: float = 1.0) -> None:
    last_error = None
    for _ in range(retries):
        try:
            pc.list_indexes()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay)
    raise RuntimeError(f"pinecone-local not ready after {retries * delay:.0f}s") from last_error


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    if pc.has_index(index_name):
        return
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        deletion_protection="disabled",
    )


def normalize_record(record: dict) -> dict:
    metadata = {k: v for k, v in record["metadata"].items() if k not in JUNK_METADATA_KEYS}
    source_path = metadata.get("source", "")
    basename = Path(source_path).name
    try:
        company = COMPANY_BY_FILE[basename]
    except KeyError as exc:
        raise KeyError(f"unrecognized source file in vector metadata: {basename!r}") from exc
    metadata["source"] = basename
    metadata["company"] = company
    return {"id": record["id"], "values": record["values"], "metadata": metadata}


def iter_batches(path: Path, batch_size: int):
    batch = []
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(normalize_record(json.loads(line)))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def assert_final_count(index, expected: int, retries: int = 15, delay: float = 1.0) -> None:
    count = -1
    for _ in range(retries):
        count = index.describe_index_stats().total_vector_count
        if count == expected:
            print(f"OK: vector count assertion passed ({count} == {expected})")
            return
        time.sleep(delay)
    raise AssertionError(f"expected {expected} vectors, found {count} after waiting")


def main() -> None:
    settings = get_settings()

    pc = Pinecone(api_key="pclocal", host=settings.pinecone_host)
    wait_until_ready(pc)

    ensure_index(pc, settings.pinecone_index, settings.embed_dimensions)
    host = resolve_index_host(pc, settings.pinecone_index)
    index = pc.Index(host=host)

    total_upserted = 0
    for batch in iter_batches(VECTORS_PATH, BATCH_SIZE):
        index.upsert(vectors=batch)
        total_upserted += len(batch)
        print(f"upserted {total_upserted} vectors so far...")

    print(f"done: {total_upserted} vectors upserted from {VECTORS_PATH.name}")
    assert_final_count(index, EXPECTED_COUNT)


if __name__ == "__main__":
    main()
