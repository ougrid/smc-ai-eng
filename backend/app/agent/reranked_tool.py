"""Reranking: cuts a wide, hybrid-retrieved candidate pool (dense + lexical,
fused via RRF -- see agent/hybrid_tool.py) down to a small, precision-ranked
final set via a cross-encoder (agent/reranker.py).

Wraps any `VectorQueryable` (in practice a `HybridTool` configured with a
wide `rerank_pool_size`, see main.py), so `nodes/vector_retrieve.py` needs
no changes -- same pattern `HybridTool` itself already established.

Reranking runs **per company**, never over the pooled cross-company set --
the rest of this pipeline never does a single global top-k specifically
because per-company chunk counts are heavily skewed (Meta 1568 vs Apple 604,
see agent/vector_tool.py), and reranking globally would let one company's
merely-good chunks crowd out another's best ones.

Fail-open, not fail-closed: retrieval-quality features in this codebase
degrade gracefully rather than crash the node (see text_search_tool.py's
DB-error handling) -- if the cross-encoder raises (e.g. the model failed to
load), this falls back to the inner tool's own ordering. It still enforces
the smaller `top_k` cut either way, so a reranker outage can't silently
balloon the evidence pool back up to `rerank_pool_size`.
"""

from app.agent.reranker import Reranker
from app.agent.vector_tool import Chunk, RejectedChunk, VectorQueryable, VectorResult


def _rerank_company(reranker: Reranker, question: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    scores = reranker.score(question, [c.text for c in chunks])
    return sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)


class RerankedTool:
    def __init__(self, inner: VectorQueryable, reranker: Reranker, *, top_k: int = 6):
        self._inner = inner
        self._reranker = reranker
        self._top_k = top_k

    def query(self, question: str, companies: list[str]) -> VectorResult:
        result = self._inner.query(question, companies)
        if not result.chunks or not companies:
            return result

        kept: list[Chunk] = []
        cut: list[RejectedChunk] = []
        for company in companies:
            company_chunks = [c for c in result.chunks if c.company == company]
            if not company_chunks:
                continue

            try:
                ranked = _rerank_company(self._reranker, question, company_chunks)
            except Exception:  # pragma: no cover -- defensive, model-error path (see module docstring)
                ranked = [(c, c.score) for c in company_chunks]

            keep, drop = ranked[: self._top_k], ranked[self._top_k :]
            kept.extend(
                Chunk(id=c.id, company=c.company, source=c.source, page=c.page, text=c.text, score=round(float(score), 6))
                for c, score in keep
            )
            cut.extend(
                RejectedChunk(id=c.id, company=c.company, score=round(float(score), 6), reason="reranked_out")
                for c, score in drop
            )

        return VectorResult(chunks=kept, rejected=result.rejected + cut)
