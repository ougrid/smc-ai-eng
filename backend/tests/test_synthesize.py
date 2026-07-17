"""synthesize node tests -- offline, stubbing the structured-output-bound
LLM runnable directly (same pattern as test_router.py).
"""

from app.agent.nodes.synthesize import build_synthesize_node
from app.agent.schemas import Citation, SynthesisEnvelope


class _StubSynthLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        return self._responses.pop(0)


def _envelope(**overrides):
    defaults = dict(reasoning="r", answer="Apple's net income was $112,010M in 2025.", citations=[])
    defaults.update(overrides)
    return SynthesisEnvelope(**defaults)


def _ok(envelope):
    return {"raw": None, "parsed": envelope, "parsing_error": None}


def _failure(error=None):
    return {"raw": None, "parsed": None, "parsing_error": error}


def test_no_evidence_refuses_without_calling_the_llm():
    llm = _StubSynthLLM([])  # would raise IndexError if ever called
    node = build_synthesize_node(llm)
    result = node({"question": "q", "sql_rows": [], "chunks": []}, {})
    assert result["envelope"] is None
    assert "don't have grounded data" in result["final_answer"]
    assert llm.calls == 0


def test_sql_evidence_produces_answer_from_envelope():
    llm = _StubSynthLLM([_ok(_envelope())])
    node = build_synthesize_node(llm)
    result = node(
        {"question": "q", "sql_rows": [{"company": "Apple", "year": 2025, "net_income": 112010000000}]},
        {},
    )
    assert result["final_answer"] == "Apple's net income was $112,010M in 2025."
    assert result["envelope"]["citations"] == []


def test_chunk_evidence_carries_citations_through():
    envelope = _envelope(
        answer="Meta grew due to ad revenue.",
        citations=[Citation(kind="chunk", source="Meta_10K.pdf", page=5, quote="ad revenue grew")],
    )
    llm = _StubSynthLLM([_ok(envelope)])
    node = build_synthesize_node(llm)
    result = node(
        {
            "question": "why did Meta grow?",
            "sql_rows": [],
            "chunks": [{"id": "c1", "company": "Meta", "source": "Meta_10K.pdf", "page": 5, "text": "ad revenue grew", "score": 0.8}],
        },
        {},
    )
    assert result["final_answer"] == "Meta grew due to ad revenue."
    assert result["envelope"]["citations"][0]["source"] == "Meta_10K.pdf"


def test_reasks_once_then_succeeds():
    llm = _StubSynthLLM([_failure(ValueError("bad")), _ok(_envelope())])
    node = build_synthesize_node(llm)
    result = node({"question": "q", "sql_rows": [{"company": "Apple", "year": 2025}]}, {})
    assert result["final_answer"] == "Apple's net income was $112,010M in 2025."
    assert llm.calls == 2


def test_fails_closed_after_second_malformed_response():
    llm = _StubSynthLLM([_failure(ValueError("bad")), _failure(ValueError("still bad"))])
    node = build_synthesize_node(llm)
    result = node({"question": "q", "sql_rows": [{"company": "Apple", "year": 2025}]}, {})
    assert result["envelope"] is None
    assert "couldn't process" in result["final_answer"]
    assert llm.calls == 2
