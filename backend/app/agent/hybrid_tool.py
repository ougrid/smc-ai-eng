"""Hybrid retrieval: fuses VectorTool's dense (Pinecone) ranking with
TextSearchTool's lexical (Postgres full-text) ranking via reciprocal rank
fusion (agent/fusion.py), before handing the result to `nodes/vector_retrieve.py`
exactly as VectorTool alone would -- same `VectorResult` shape, same
`chunks`/`rejected` fields, so nothing downstream needs to know fusion
happened (see `VectorQueryable` in agent/vector_tool.py).

Chunk ids are shared between the two systems: both are loaded from the
same data/pinecone_vectors.jsonl.gz source (Pinecone via load_pinecone.py,
chunk_text via load_chunk_text.py), so fusion is a simple per-company join
by id -- no separate id-mapping layer needed.

`score` on a returned Chunk is the fused RRF score once fusion has run
(not the original cosine similarity or `ts_rank_cd` value -- neither is
meaningful once merged with the other). `score_floor`/boilerplate/duplicate
filtering on the dense side already happened inside VectorTool.query by the
time this module sees its output; this module's own cut is "top `top_k`
per company after fusion", recorded in `rejected` as reason "fusion_cut".
"""

from app.agent.fusion import reciprocal_rank_fusion
from app.agent.text_search_tool import TextHit, TextSearchTool
from app.agent.vector_tool import (
    Chunk,
    RejectedChunk,
    VectorResult,
    VectorTool,
    dedupe_chunks,
    is_boilerplate,
)


class HybridTool:
    def __init__(self, vector_tool: VectorTool, text_tool: TextSearchTool | None, *, top_k: int = 10):
        self._vector_tool = vector_tool
        self._text_tool = text_tool
        self._top_k = top_k

    def query(self, question: str, companies: list[str]) -> VectorResult:
        dense = self._vector_tool.query(question, companies)
        if not companies or self._text_tool is None:
            return dense

        text_hits = self._text_tool.query(question, companies)
        if not text_hits:
            return dense

        dense_by_id: dict[str, Chunk] = {c.id: c for c in dense.chunks}
        text_by_id: dict[str, Chunk] = {}
        for hit in text_hits:
            if hit.id in dense_by_id or hit.id in text_by_id:
                continue
            if is_boilerplate(hit.text):  # dense-only filter never saw this candidate
                continue
            text_by_id[hit.id] = _chunk_from_hit(hit)

        kept: list[Chunk] = []
        fusion_cut: list[RejectedChunk] = []
        for company in companies:
            dense_ids = [c.id for c in dense.chunks if c.company == company]
            text_ids = [h.id for h in text_hits if h.company == company]
            if not dense_ids and not text_ids:
                continue

            rrf_scores = reciprocal_rank_fusion(dense_ids, text_ids)
            pool: dict[str, Chunk] = {i: dense_by_id[i] for i in dense_ids}
            pool.update({i: text_by_id[i] for i in text_ids if i in text_by_id})
            ranked = sorted(pool.values(), key=lambda c: rrf_scores[c.id], reverse=True)

            keep, cut = ranked[: self._top_k], ranked[self._top_k :]
            kept.extend(_rescored(c, rrf_scores[c.id]) for c in keep)
            fusion_cut.extend(
                RejectedChunk(id=c.id, company=c.company, score=round(rrf_scores[c.id], 6), reason="fusion_cut")
                for c in cut
            )

        kept, duplicates = dedupe_chunks(kept)
        kept_ids = {c.id for c in kept}
        carried = [r for r in dense.rejected if r.id not in kept_ids]

        return VectorResult(chunks=kept, rejected=carried + fusion_cut + duplicates)


def _chunk_from_hit(hit: TextHit) -> Chunk:
    return Chunk(id=hit.id, company=hit.company, source=hit.source, page=hit.page, text=hit.text, score=hit.rank)


def _rescored(chunk: Chunk, score: float) -> Chunk:
    return Chunk(
        id=chunk.id, company=chunk.company, source=chunk.source, page=chunk.page, text=chunk.text,
        score=round(score, 6),
    )
