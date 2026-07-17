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
    assert tool.calls == [("why did Meta grow?", ["Meta"])]


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
    assert out["rejected_chunks"] == [{"id": "c2", "company": "Meta", "score": 0.1}]
