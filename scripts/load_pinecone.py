"""Idempotent pinecone-local seeder.

pinecone-local has no persistence across restarts, so this script must be
re-run after every `docker compose up`. Upserts are keyed by stable ids, so
re-running is safe.
"""

import gzip
import json
import os
import sys
import time

from pinecone import Pinecone, ServerlessSpec

CONTROL_PLANE = os.environ.get("PINECONE_HOST", "http://localhost:5080")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "10k-filings")
EXPECTED_COUNT = 4072
BATCH_SIZE = 200
DROP_KEYS = {"creator", "producer", "moddate", "creationdate"}

# Canonical company keys = SQL table names (deliberate: Google not Alphabet,
# Meta not Facebook -- matches financial_data.company values).
COMPANY_BY_FILE = {
    "Alphabet_10K_FY2025.pdf": "Google",
    "Amazon_10K_FY2025.pdf": "Amazon",
    "Apple_10K_FY2025.pdf": "Apple",
    "Meta_10K_FY2025.pdf": "Meta",
}


def resolve_data_plane_host(pc: Pinecone, index_name: str) -> str:
    """Resolve the data-plane host for an index. Never hardcode the port --
    pinecone-local assigns each index a port in 5081-5090."""
    host = pc.describe_index(index_name).host
    if not host.startswith("http"):
        host = f"http://{host}"
    return host


def normalize_record(record: dict) -> dict:
    metadata = {k: v for k, v in record["metadata"].items() if k not in DROP_KEYS}
    filename = os.path.basename(metadata["source"])
    metadata["source"] = filename
    metadata["company"] = COMPANY_BY_FILE[filename]
    return {"id": record["id"], "values": record["values"], "metadata": metadata}


def main() -> None:
    pc = Pinecone(api_key="pclocal", host=CONTROL_PLANE)

    for attempt in range(30):
        try:
            pc.list_indexes()
            break
        except Exception:
            time.sleep(1)
    else:
        sys.exit(f"pinecone-local not reachable at {CONTROL_PLANE}")

    if not pc.has_index(INDEX_NAME):
        pc.create_index(
            name=INDEX_NAME,
            dimension=512,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
        )

    host = resolve_data_plane_host(pc, INDEX_NAME)
    index = pc.Index(host=host)

    total = 0
    batch = []
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "pinecone_vectors.jsonl.gz",
    )
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            batch.append(normalize_record(json.loads(line)))
            if len(batch) == BATCH_SIZE:
                index.upsert(vectors=batch)
                total += len(batch)
                batch = []
        if batch:
            index.upsert(vectors=batch)
            total += len(batch)

    count = 0
    for _ in range(15):
        count = index.describe_index_stats().total_vector_count
        if count == EXPECTED_COUNT:
            break
        time.sleep(1)

    assert count == EXPECTED_COUNT, f"expected {EXPECTED_COUNT} vectors, index has {count}"
    print(f"OK: {count} vectors in '{INDEX_NAME}' at {host} (upserted {total} records)")


if __name__ == "__main__":
    main()
