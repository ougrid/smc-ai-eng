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


def test_language_directive_uses_route_detected_language_not_question_text():
    captured = {}

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured["messages"] = messages
            return _ok(_envelope(answer="answer"))

    node = build_synthesize_node(_CapturingLLM())
    node(
        {
            "question": "What was Apple's net income from 2022 to 2025?",
            "sql_rows": [{"company": "Apple", "year": 2025}],
            "route": {"language": "en"},
        },
        {},
    )
    human_message = captured["messages"][1][1]
    assert 'TARGET LANGUAGE: English ("en")' in human_message


def test_language_directive_defaults_to_english_when_route_missing():
    captured = {}

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured["messages"] = messages
            return _ok(_envelope(answer="answer"))

    node = build_synthesize_node(_CapturingLLM())
    node({"question": "q", "sql_rows": [{"company": "Apple", "year": 2025}]}, {})
    human_message = captured["messages"][1][1]
    assert 'TARGET LANGUAGE: English ("en")' in human_message


def test_history_is_threaded_between_system_prompt_and_question():
    captured = {}

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured["messages"] = messages
            return _ok(_envelope(answer="answer"))

    node = build_synthesize_node(_CapturingLLM())
    node(
        {
            "question": "give me insights",
            "sql_rows": [{"company": "Amazon", "year": 2025}],
            "history": [("user", "AMZN"), ("assistant", "What about it?")],
        },
        {},
    )
    messages = captured["messages"]
    assert messages[1] == ("human", "AMZN")
    assert messages[2] == ("ai", "What about it?")
    assert messages[3][0] == "human"
    assert "give me insights" in messages[3][1]


def test_absent_history_leaves_message_shape_unchanged():
    captured = {}

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured["messages"] = messages
            return _ok(_envelope(answer="answer"))

    node = build_synthesize_node(_CapturingLLM())
    node({"question": "q", "sql_rows": [{"company": "Apple", "year": 2025}]}, {})
    assert len(captured["messages"]) == 2  # system + human, exactly as before this feature


def test_verify_retry_hint_reaches_the_llm_prompt():
    captured = {}

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured["messages"] = messages
            return _ok(_envelope(answer="corrected"))

    node = build_synthesize_node(_CapturingLLM())
    node(
        {
            "question": "q",
            "sql_rows": [{"company": "Apple", "year": 2025}],
            "verify": {"ok": False, "ungrounded": ["999,999"], "attempt": 1},
        },
        {},
    )
    human_message = captured["messages"][1][1]
    assert "999,999" in human_message
    assert "could not be verified" in human_message


def test_dangling_citation_retry_hint_reaches_the_llm_prompt():
    captured = {}

    class _CapturingLLM:
        def invoke(self, messages, config=None):
            captured["messages"] = messages
            return _ok(_envelope(answer="corrected"))

    node = build_synthesize_node(_CapturingLLM())
    node(
        {
            "question": "q",
            "sql_rows": [{"company": "Apple", "year": 2025}],
            "verify": {
                "ok": False,
                "ungrounded": [],
                "dangling_citations": ["[Meta_10K.pdf, p.99]"],
                "attempt": 1,
            },
        },
        {},
    )
    human_message = captured["messages"][1][1]
    assert "[Meta_10K.pdf, p.99]" in human_message
    assert "did not match any" in human_message
