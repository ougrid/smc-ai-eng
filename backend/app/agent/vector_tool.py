"""Vector retrieval: one embedding call, then one Pinecone query per
company -- never a single global top-k. Per-company chunk counts are
heavily skewed (Meta 1568 vs Apple 604, see docs/implementation-plan.md
verified data facts), so a global top-k would starve the smaller filings.

Chunks scoring below `score_floor` are dropped from `chunks` but still
recorded in `rejected`, which flows straight into the `debug` payload --
this is what turns Day-3 score-floor tuning into reading a JSON field
instead of re-running queries by hand.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


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


@dataclass
class VectorResult:
    chunks: list[Chunk] = field(default_factory=list)
    rejected: list[RejectedChunk] = field(default_factory=list)


def _matches_of(response: Any) -> list[Any]:
    return response["matches"] if isinstance(response, dict) else response.matches


def _field(match: Any, name: str, default: Any = None) -> Any:
    if isinstance(match, dict):
        return match.get(name, default)
    return getattr(match, name, default)


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
                    rejected.append(RejectedChunk(id=match_id, company=company, score=score))
                    continue
                metadata = _field(match, "metadata", {}) or {}
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

        return VectorResult(chunks=chunks, rejected=rejected)
