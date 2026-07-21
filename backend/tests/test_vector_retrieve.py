from app.agent.nodes.vector_retrieve import build_vector_retrieve_node
from app.agent.vector_tool import Chunk, RejectedChunk, VectorResult


class _StubVectorTool:
    def __init__(self, result: VectorResult):
        self._result = result
        self.calls = []

    def query(self, question, companies):
        self.calls.append((question, companies))
        return self._result


def test_queries_only_vector_eligible_companies():
    tool = _StubVectorTool(VectorResult())
    node = build_vector_retrieve_node(tool)
    node({"question": "why did Meta grow?", "vector_companies": ["Meta"]}, {})
    assert len(tool.calls) == 1
    query, companies = tool.calls[0]
    assert companies == ["Meta"]
    assert query.startswith("why did Meta grow?")


def test_reformulates_query_with_english_narrative_hint():
    tool = _StubVectorTool(VectorResult())
    node = build_vector_retrieve_node(tool)
    node({"question": "ทำไมรายได้ถึงโต", "vector_companies": ["Meta"], "metrics": ["revenue"]}, {})
    query, _ = tool.calls[0]
    assert "ทำไมรายได้ถึงโต" in query
    assert "revenue" in query
    assert "business strategy" in query


def test_reformulates_with_generic_hint_when_no_metrics():
    tool = _StubVectorTool(VectorResult())
    node = build_vector_retrieve_node(tool)
    node({"question": "compare strategy", "vector_companies": ["Meta"]}, {})
    query, _ = tool.calls[0]
    assert "business strategy" in query


def test_zero_kept_chunks_for_a_vector_eligible_company_adds_coverage_note():
    tool = _StubVectorTool(VectorResult())  # no chunks kept for either company
    node = build_vector_retrieve_node(tool)
    out = node({"question": "why did they grow?", "vector_companies": ["Apple", "Meta"]}, {})
    assert len(out["coverage_notes"]) == 2
    assert any("Apple" in n for n in out["coverage_notes"])
    assert any("Meta" in n for n in out["coverage_notes"])


def test_covered_company_gets_no_gap_note_but_uncovered_one_does():
    result = VectorResult(
        chunks=[Chunk(id="c1", company="Meta", source="Meta_10K.pdf", page=5, text="...", score=0.8)]
    )
    tool = _StubVectorTool(result)
    node = build_vector_retrieve_node(tool)
    out = node({"question": "why did they grow?", "vector_companies": ["Apple", "Meta"]}, {})
    assert len(out["coverage_notes"]) == 1
    assert "Apple" in out["coverage_notes"][0]


def test_existing_coverage_notes_are_preserved_not_overwritten():
    tool = _StubVectorTool(VectorResult())
    node = build_vector_retrieve_node(tool)
    out = node(
        {
            "question": "q",
            "vector_companies": ["Apple"],
            "coverage_notes": ["Microsoft has no 10-K indexed"],
        },
        {},
    )
    assert "Microsoft has no 10-K indexed" in out["coverage_notes"]
    assert any("Apple" in n for n in out["coverage_notes"])


def test_serializes_chunks_and_rejected_into_plain_dicts():
    result = VectorResult(
        chunks=[Chunk(id="c1", company="Meta", source="Meta_10K.pdf", page=5, text="...", score=0.8)],
        rejected=[RejectedChunk(id="c2", company="Meta", score=0.1)],
    )
    node = build_vector_retrieve_node(_StubVectorTool(result))
    out = node({"question": "q", "vector_companies": ["Meta"]}, {})
    assert out["chunks"] == [
        {"id": "c1", "company": "Meta", "source": "Meta_10K.pdf", "page": 5, "text": "...", "score": 0.8}
    ]
    assert out["rejected_chunks"] == [
        {"id": "c2", "company": "Meta", "score": 0.1, "reason": "below_floor"}
    ]
