"""RerankedTool tests -- offline, stubbing both the inner VectorQueryable
and the Reranker protocol so no real model or live retrieval is involved.
"""

from app.agent.reranked_tool import RerankedTool
from app.agent.vector_tool import Chunk, RejectedChunk, VectorResult


class _StubInner:
    def __init__(self, result: VectorResult):
        self._result = result

    def query(self, question, companies):
        return self._result


class _StubReranker:
    """Scores by an explicit {text: score} map, so ordering is fully
    controllable per test rather than depending on chunk content."""

    def __init__(self, scores_by_text: dict[str, float]):
        self._scores_by_text = scores_by_text
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, question, texts):
        self.calls.append((question, list(texts)))
        return [self._scores_by_text[t] for t in texts]


class _RaisingReranker:
    def score(self, question, texts):
        raise RuntimeError("model failed to load")


def _chunk(id_, company, text, score=0.5):
    return Chunk(id=id_, company=company, source="Meta_10K.pdf", page=5, text=text, score=score)


def test_no_chunks_passes_result_through_unchanged():
    result = VectorResult()
    tool = RerankedTool(_StubInner(result), _StubReranker({}), top_k=2)
    assert tool.query("q", ["Meta"]) is result


def test_no_companies_passes_result_through_unchanged():
    result = VectorResult(chunks=[_chunk("c1", "Meta", "a")])
    tool = RerankedTool(_StubInner(result), _StubReranker({"a": 1.0}), top_k=2)
    assert tool.query("q", []) is result


def test_reranker_reorders_and_cuts_to_top_k():
    result = VectorResult(
        chunks=[_chunk("c1", "Meta", "low relevance"), _chunk("c2", "Meta", "high relevance")]
    )
    reranker = _StubReranker({"low relevance": 0.1, "high relevance": 0.9})
    tool = RerankedTool(_StubInner(result), reranker, top_k=1)
    out = tool.query("q", ["Meta"])
    assert [c.id for c in out.chunks] == ["c2"]
    assert out.chunks[0].score == 0.9


def test_cut_chunks_are_rejected_with_reranked_out_reason():
    result = VectorResult(
        chunks=[_chunk("c1", "Meta", "low"), _chunk("c2", "Meta", "high")]
    )
    reranker = _StubReranker({"low": 0.1, "high": 0.9})
    tool = RerankedTool(_StubInner(result), reranker, top_k=1)
    out = tool.query("q", ["Meta"])
    assert len(out.rejected) == 1
    assert out.rejected[0].id == "c1"
    assert out.rejected[0].reason == "reranked_out"
    assert out.rejected[0].score == 0.1


def test_existing_rejected_reasons_are_carried_through():
    result = VectorResult(
        chunks=[_chunk("c1", "Meta", "a")],
        rejected=[RejectedChunk(id="r1", company="Meta", score=0.1, reason="fusion_cut")],
    )
    tool = RerankedTool(_StubInner(result), _StubReranker({"a": 0.5}), top_k=5)
    out = tool.query("q", ["Meta"])
    assert any(r.id == "r1" and r.reason == "fusion_cut" for r in out.rejected)


def test_companies_are_reranked_independently():
    result = VectorResult(
        chunks=[
            _chunk("a1", "Apple", "apple text"),
            _chunk("m1", "Meta", "meta text"),
        ]
    )
    reranker = _StubReranker({"apple text": 0.9, "meta text": 0.9})
    tool = RerankedTool(_StubInner(result), reranker, top_k=1)
    out = tool.query("q", ["Apple", "Meta"])
    assert {c.company for c in out.chunks} == {"Apple", "Meta"}
    # each company's single chunk survives its own top_k=1 cut independently
    assert len(out.chunks) == 2


def test_reranker_failure_falls_back_to_inner_ordering_but_still_cuts():
    result = VectorResult(
        chunks=[
            _chunk("c1", "Meta", "first", score=0.9),
            _chunk("c2", "Meta", "second", score=0.5),
        ]
    )
    tool = RerankedTool(_StubInner(result), _RaisingReranker(), top_k=1)
    out = tool.query("q", ["Meta"])
    assert len(out.chunks) == 1
    assert out.chunks[0].id == "c1"  # inner tool's own (higher) score wins the fallback ordering
    assert out.rejected[0].reason == "reranked_out"


def test_reranker_is_called_once_per_company_with_that_companys_texts():
    result = VectorResult(
        chunks=[_chunk("a1", "Apple", "apple text"), _chunk("m1", "Meta", "meta text")]
    )
    reranker = _StubReranker({"apple text": 0.5, "meta text": 0.5})
    tool = RerankedTool(_StubInner(result), reranker, top_k=5)
    tool.query("q", ["Apple", "Meta"])
    assert reranker.calls == [("q", ["apple text"]), ("q", ["meta text"])]
