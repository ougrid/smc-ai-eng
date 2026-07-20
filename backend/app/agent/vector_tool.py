"""Vector retrieval: one embedding call, then one Pinecone query per
company -- never a single global top-k. Per-company chunk counts are
heavily skewed (Meta 1568 vs Apple 604, see docs/implementation-plan.md
verified data facts), so a global top-k would starve the smaller filings.

Chunks scoring below `score_floor`, dominated by print-to-PDF header noise
(page-header lines like "4/20/26, 12:05 PM goog-20251231 file:///...", see
docs/implementation-plan.md verified data facts), or duplicating another
kept chunk's exact text, are dropped from `chunks` but still recorded in
`rejected` with a `reason`, which flows straight into the `debug` payload --
this is what turns score-floor/boilerplate tuning into reading a JSON field
instead of re-running queries by hand.

Duplicate detection exists because the *provided* source file
(data/pinecone_vectors.jsonl.gz, loaded as-is per CLAUDE.md -- never
regenerated or re-embedded) turns out to contain each chunk of real content
roughly twice, under different ids. Pinecone itself has no dedup, so both
copies can surface in a query's top-k; this module dedupes by (company,
text) at read time instead, keeping the highest-scoring copy -- the index
still holds and reports all 4072 loaded vectors unchanged.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

_HEADER_NOISE = re.compile(
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s*\d{1,2}:\d{2}\s*[AP]M.*?file:///", re.IGNORECASE
)
_MIN_SUBSTANTIVE_CHARS = 40  # below this, after stripping header noise, there's nothing left to cite


def is_boilerplate(text: str) -> bool:
    """Public (not `_`-prefixed): also used by agent/hybrid_tool.py to
    filter text-search-only candidates, which never pass through
    VectorTool.query's own boilerplate check."""
    if not _HEADER_NOISE.search(text):
        return False  # short-but-real text (no header noise present) is not our concern here
    return len(_HEADER_NOISE.sub("", text).strip()) < _MIN_SUBSTANTIVE_CHARS


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass
class Chunk:
    id: str
    company: str
    source: str | None
    page: int | None
    text: str
    score: float


@dataclass
class RejectedChunk:
    id: str
    company: str
    score: float
    reason: str = "below_floor"  # "below_floor" | "boilerplate" | "duplicate" | "fusion_cut"


@dataclass
class VectorResult:
    chunks: list[Chunk] = field(default_factory=list)
    rejected: list[RejectedChunk] = field(default_factory=list)


class VectorQueryable(Protocol):
    """Structural interface `nodes/vector_retrieve.py` and `graph.py` depend
    on -- satisfied by both `VectorTool` (dense-only) and
    `agent/hybrid_tool.py`'s `HybridTool` (dense+lexical fused), so the node
    never needs to know which one it was handed."""

    def query(self, question: str, companies: list[str]) -> VectorResult: ...


def _matches_of(response: Any) -> list[Any]:
    return response["matches"] if isinstance(response, dict) else response.matches


def _field(match: Any, name: str, default: Any = None) -> Any:
    if isinstance(match, dict):
        return match.get(name, default)
    return getattr(match, name, default)


def dedupe_chunks(chunks: list[Chunk]) -> tuple[list[Chunk], list[RejectedChunk]]:
    best: dict[tuple[str, str], Chunk] = {}
    for chunk in chunks:
        key = (chunk.company, chunk.text)
        if key not in best or chunk.score > best[key].score:
            best[key] = chunk

    kept_ids = {c.id for c in best.values()}
    kept = [c for c in chunks if c.id in kept_ids]
    dropped = [
        RejectedChunk(id=c.id, company=c.company, score=c.score, reason="duplicate")
        for c in chunks
        if c.id not in kept_ids
    ]
    return kept, dropped


class VectorTool:
    def __init__(
        self,
        index: Any,
        embed: Embedder,
        *,
        top_k: int = 6,
        score_floor: float = 0.25,
    ):
        self._index = index
        self._embed = embed
        self._top_k = top_k
        self._score_floor = score_floor

    def query(self, question: str, companies: list[str]) -> VectorResult:
        if not companies:
            return VectorResult()

        vector = self._embed.embed_query(question)
        chunks: list[Chunk] = []
        rejected: list[RejectedChunk] = []

        for company in companies:
            response = self._index.query(
                vector=vector,
                top_k=self._top_k,
                filter={"company": {"$eq": company}},
                include_metadata=True,
            )
            for match in _matches_of(response):
                match_id = _field(match, "id")
                score = _field(match, "score", 0.0)
                if score < self._score_floor:
                    rejected.append(
                        RejectedChunk(id=match_id, company=company, score=score, reason="below_floor")
                    )
                    continue
                metadata = _field(match, "metadata", {}) or {}
                text = metadata.get("text", "")
                if is_boilerplate(text):
                    rejected.append(
                        RejectedChunk(id=match_id, company=company, score=score, reason="boilerplate")
                    )
                    continue
                chunks.append(
                    Chunk(
                        id=match_id,
                        company=company,
                        source=metadata.get("source"),
                        page=metadata.get("page"),
                        text=metadata.get("text", ""),
                        score=score,
                    )
                )

        chunks, duplicates = dedupe_chunks(chunks)
        rejected.extend(duplicates)

        return VectorResult(chunks=chunks, rejected=rejected)
