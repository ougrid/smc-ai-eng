-- chunk_text: lexical (full-text search) half of hybrid retrieval,
-- populated by scripts/load_chunk_text.py from the same
-- data/pinecone_vectors.jsonl.gz source Pinecone is seeded from. Rows share
-- ids with the Pinecone vectors, so backend/app/agent/hybrid_tool.py can
-- fuse dense + lexical results by id (see agent/text_search_tool.py).
CREATE TABLE chunk_text (
    id TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    source TEXT,
    page INTEGER,
    text TEXT NOT NULL,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX chunk_text_tsv_idx ON chunk_text USING GIN (tsv);
CREATE INDEX chunk_text_company_idx ON chunk_text (company);
