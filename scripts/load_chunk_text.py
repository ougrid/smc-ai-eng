"""Idempotent Postgres full-text loader for `chunk_text` -- the lexical
half of hybrid retrieval (see backend/app/agent/text_search_tool.py and
hybrid_tool.py). Reads the same data/pinecone_vectors.jsonl.gz source
scripts/load_pinecone.py seeds Pinecone from (never regenerated or
re-embedded, per CLAUDE.md's "load data as-is"), inserting the same ids so
dense (Pinecone) and lexical (Postgres) hits fuse by id downstream.

Unlike pinecone-local, Postgres persists across `docker compose up` (no
-v), but the `chunk_text` table itself is only created by the fresh-volume
initdb script (scripts/initdb/02_chunk_text.sql) -- so on a Postgres volume
created before this feature existed, `down -v` is needed once to pick up
the new table (same "-v is load-bearing" caveat as the rest of this repo's
initdb scripts). This loader itself is idempotent (`ON CONFLICT (id) DO
NOTHING`) and safe to re-run every `make seed`, same as load_pinecone.py.
"""

import gzip
import json
import os

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://app:app_local_dev@localhost:5432/findata"
)
EXPECTED_COUNT = 4072
BATCH_SIZE = 500

# Mirrors scripts/load_pinecone.py's COMPANY_BY_FILE -- canonical company
# keys are the SQL table's own names (Google, Meta, not Alphabet/Facebook).
COMPANY_BY_FILE = {
    "Alphabet_10K_FY2025.pdf": "Google",
    "Amazon_10K_FY2025.pdf": "Amazon",
    "Apple_10K_FY2025.pdf": "Apple",
    "Meta_10K_FY2025.pdf": "Meta",
}

_INSERT = text(
    """
    INSERT INTO chunk_text (id, company, source, page, text)
    VALUES (:id, :company, :source, :page, :text)
    ON CONFLICT (id) DO NOTHING
    """
)


def _rows():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "pinecone_vectors.jsonl.gz",
    )
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata = record["metadata"]
            filename = os.path.basename(metadata["source"])
            yield {
                "id": record["id"],
                "company": COMPANY_BY_FILE[filename],
                "source": filename,
                "page": metadata.get("page"),
                "text": metadata.get("text", ""),
            }


def main() -> None:
    engine = create_engine(DATABASE_URL)
    total = 0
    batch = []
    with engine.begin() as conn:
        for row in _rows():
            batch.append(row)
            if len(batch) == BATCH_SIZE:
                conn.execute(_INSERT, batch)
                total += len(batch)
                batch = []
        if batch:
            conn.execute(_INSERT, batch)
            total += len(batch)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM chunk_text")).scalar_one()

    assert count == EXPECTED_COUNT, f"expected {EXPECTED_COUNT} chunk_text rows, got {count}"
    print(f"OK: {count} rows in 'chunk_text' (processed {total} records this run)")


if __name__ == "__main__":
    main()
