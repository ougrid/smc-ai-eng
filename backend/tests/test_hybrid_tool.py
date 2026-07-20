"""HybridTool tests -- offline, stubbing both VectorTool and TextSearchTool
so no live Pinecone/Postgres is involved. Focuses on the fusion/cut/carry
logic; individual dense- and lexical-side filtering is already covered by
test_vector_tool.py and test_text_search_tool.py respectively.
"""

from app.agent.hybrid_tool import HybridTool
from app.agent.text_search_tool import TextHit
from app.agent.vector_tool import Chunk, RejectedChunk, VectorResult


class _StubVectorTool:
    def __init__(self, result):
        self._result = result

    def query(self, question, companies):
        return self._result


class _StubTextTool:
    def __init__(self, hits):
        self._hits = hits

    def query(self, question, companies):
        return self._hits


def _chunk(id_, company, score, text="dense text"):
    return Chunk(id=id_, company=company, source="Meta_10K.pdf", page=5, text=text, score=score)


def _hit(id_, company, rank, text="lexical text"):
    return TextHit(id=id_, company=company, source="Meta_10K.pdf", page=7, rank=rank, text=text)


def test_no_text_hits_passes_dense_result_through_unchanged():
    dense = VectorResult(chunks=[_chunk("d1", "Meta", 0.8)])
    result = HybridTool(_StubVectorTool(dense), _StubTextTool([])).query("q", ["Meta"])
    assert result is dense


def test_no_text_tool_passes_dense_result_through_unchanged():
    dense = VectorResult(chunks=[_chunk("d1", "Meta", 0.8)])
    result = HybridTool(_StubVectorTool(dense), None).query("q", ["Meta"])
    assert result is dense


def test_no_companies_passes_dense_result_through_unchanged():
    dense = VectorResult()
    hits = [_hit("t1", "Meta", 0.9)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits)).query("q", [])
    assert result is dense


def test_text_only_hit_is_added_alongside_dense_hits():
    dense = VectorResult(chunks=[_chunk("d1", "Meta", 0.9)])
    hits = [_hit("t1", "Meta", 0.5)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits), top_k=10).query("q", ["Meta"])
    assert {c.id for c in result.chunks} == {"d1", "t1"}


def test_boilerplate_text_only_hit_is_dropped_not_added():
    noisy = "4/20/26, 12:05 PM goog-20251231 file:///Users/x/goog.htm"
    dense = VectorResult()
    hits = [_hit("t1", "Meta", 0.9, text=noisy)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits)).query("q", ["Meta"])
    assert result.chunks == []


def test_fusion_cuts_lowest_ranked_candidates_beyond_top_k():
    dense = VectorResult(chunks=[_chunk("d1", "Meta", 0.9), _chunk("d2", "Meta", 0.8)])
    hits = [_hit("t1", "Meta", 0.95), _hit("t2", "Meta", 0.85)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits), top_k=2).query("q", ["Meta"])
    assert len(result.chunks) == 2
    assert any(r.reason == "fusion_cut" for r in result.rejected)


def test_dense_rejected_reasons_are_carried_through_when_fusion_runs():
    dense = VectorResult(
        chunks=[_chunk("d1", "Meta", 0.9)],
        rejected=[RejectedChunk(id="r1", company="Meta", score=0.1, reason="below_floor")],
    )
    hits = [_hit("t1", "Meta", 0.5)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits)).query("q", ["Meta"])
    assert any(r.id == "r1" and r.reason == "below_floor" for r in result.rejected)


def test_fused_chunk_score_is_rrf_not_the_original_dense_score():
    dense = VectorResult(chunks=[_chunk("d1", "Meta", 0.9)])
    hits = [_hit("t1", "Meta", 0.5)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits)).query("q", ["Meta"])
    d1 = next(c for c in result.chunks if c.id == "d1")
    assert 0 < d1.score < 1
    assert d1.score != 0.9


def test_companies_are_fused_independently():
    dense = VectorResult(chunks=[_chunk("d1", "Apple", 0.9), _chunk("d2", "Meta", 0.9)])
    hits = [_hit("t1", "Apple", 0.5)]
    result = HybridTool(_StubVectorTool(dense), _StubTextTool(hits), top_k=10).query(
        "q", ["Apple", "Meta"]
    )
    ids_by_company = {(c.company, c.id) for c in result.chunks}
    assert ("Apple", "t1") in ids_by_company
    assert ("Meta", "t1") not in ids_by_company
