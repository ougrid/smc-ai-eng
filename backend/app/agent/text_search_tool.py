"""Lexical half of hybrid retrieval: Postgres full-text search over the
`chunk_text` table (populated by scripts/load_chunk_text.py from the same
data/pinecone_vectors.jsonl.gz source Pinecone is seeded from -- same ids,
so results fuse with VectorTool's dense hits by id in agent/hybrid_tool.py).

Dense embeddings alone under-rank exact-term matches -- ticker symbols,
section labels like "Item 1A", named products -- that cosine similarity
treats as just another word among many; full-text search finds those
directly. This is the retrieval-quality gap named in
docs/implementation-plan.md's Future-improvements roadmap.

Queries via the `agent_ro` engine, same as SqlTool -- this path never
receives LLM-generated SQL (the query is fixed, only the search term and
company filter are parameterized), but it still has no reason to hold
write access.

**Why an OR-combined `to_tsquery`, not `plainto_tsquery`/`websearch_to_tsquery`**:
both of those AND every significant word together by default, so a
natural-language question of even moderate length ("What does Meta's 10-K
say in Item 1A about risk factors related to competition?") almost never
matches a single chunk that happens to contain literally every one of
those words -- verified live: the plain question alone matched zero
`chunk_text` rows against `plainto_tsquery`. This module instead tokenizes
the question itself and builds an OR-combined query (`word1 | word2 | ...`)
via `to_tsquery`, so a chunk matching even a handful of the salient terms
surfaces; `ts_rank_cd` already rewards chunks that match more terms, so
ranking quality doesn't depend on requiring an exact whole-question match.

Fail-open, not fail-closed: a query error here (most likely `chunk_text`
not existing yet, e.g. a pre-hybrid Postgres volume that hasn't been
recreated with `docker compose down -v`) degrades to "no lexical hits" --
`HybridTool` then falls back to dense-only results, exactly like before
this feature existed -- rather than crashing the whole retrieval node.
"""

import re
from dataclasses import dataclass

from sqlalchemy import Engine, text

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_MIN_TOKEN_LEN = 2

_QUERY = text(
    """
    SELECT id, company, source, page, text,
           ts_rank_cd(tsv, to_tsquery('english', :tsq)) AS rank
    FROM chunk_text
    WHERE company = :company
      AND tsv @@ to_tsquery('english', :tsq)
    ORDER BY rank DESC
    LIMIT :top_k
    """
)


def _to_or_tsquery(question: str) -> str | None:
    """Tokens are stripped to bare alphanumerics before joining with `|` --
    to_tsquery's operator syntax (`&`, `|`, `!`, `(`, `)`, `:`) would
    otherwise turn stray punctuation in the question into a syntax error
    rather than a search term."""
    tokens = [t.lower() for t in _WORD_RE.findall(question) if len(t) >= _MIN_TOKEN_LEN]
    if not tokens:
        return None
    return " | ".join(tokens)


@dataclass
class TextHit:
    id: str
    company: str
    source: str | None
    page: int | None
    text: str
    rank: float


class TextSearchTool:
    def __init__(self, engine: Engine, *, top_k: int = 10):
        self._engine = engine
        self._top_k = top_k

    def query(self, question: str, companies: list[str]) -> list[TextHit]:
        if not companies:
            return []

        tsq = _to_or_tsquery(question)
        if tsq is None:
            return []

        hits: list[TextHit] = []
        try:
            with self._engine.connect() as conn:
                for company in companies:
                    rows = conn.execute(
                        _QUERY, {"tsq": tsq, "company": company, "top_k": self._top_k}
                    ).mappings().all()
                    hits.extend(TextHit(**dict(row)) for row in rows)
        except Exception:  # pragma: no cover -- defensive, DB-error path (see module docstring)
            return []

        return hits
