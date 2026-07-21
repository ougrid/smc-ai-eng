"""Route node tests -- offline, stubbing the structured-output-bound LLM
runnable directly (the `RouteLLM` protocol) rather than a real chat model.
"""

from app.agent.coverage import CoverageMap
from app.agent.nodes.route import SYSTEM_PROMPT, build_route_node
from app.agent.schemas import CompanyMention, RouteDecision

COVERAGE = CoverageMap(
    sql_years={
        "Apple": [2022, 2023, 2024, 2025],
        "Google": [2022, 2023, 2024, 2025],
        "Meta": [2022, 2023, 2024, 2025],
    }
)


class _StubRouteLLM:
    """Each entry in `responses` is one `invoke()` return value, in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        return self._responses.pop(0)


def _parsed(**overrides):
    defaults = dict(
        reasoning="r",
        intent="financial",
        companies=[CompanyMention(mentioned="Apple", canonical="Apple", confident=True)],
        years=[2024],
        metrics=["net_income"],
        route="sql",
        clarification=None,
        language="en",
    )
    defaults.update(overrides)
    return RouteDecision(**defaults)


def _ok(decision):
    return {"raw": None, "parsed": decision, "parsing_error": None}


def _failure(error=None):
    return {"raw": None, "parsed": None, "parsing_error": error}


def _run(llm):
    node = build_route_node(COVERAGE, llm)
    return node({"question": "irrelevant -- responses are stubbed"}, {})


def test_valid_decision_routes_to_sql():
    llm = _StubRouteLLM([_ok(_parsed())])
    result = _run(llm)
    assert result["effective_route"] == "sql"
    assert result["companies"] == ["Apple"]
    assert llm.calls == 1


def test_parsed_none_once_then_succeeds_reasks_once():
    llm = _StubRouteLLM([_failure(ValueError("bad json")), _ok(_parsed())])
    result = _run(llm)
    assert result["effective_route"] == "sql"
    assert llm.calls == 2


def test_parsed_none_twice_fails_closed_to_refuse():
    llm = _StubRouteLLM([_failure(ValueError("bad json")), _failure(ValueError("still bad"))])
    result = _run(llm)
    assert result["effective_route"] == "refuse"
    assert result["refusal_reason"] == "malformed_output"
    assert llm.calls == 2


def test_missing_parsing_error_behaves_identically_to_present_one():
    llm = _StubRouteLLM([_failure(None), _ok(_parsed())])
    result = _run(llm)
    assert result["effective_route"] == "sql"
    assert llm.calls == 2


def test_intent_off_topic_routes_to_refuse():
    llm = _StubRouteLLM(
        [_ok(_parsed(intent="off_topic", companies=[], years=[], route="refuse"))]
    )
    result = _run(llm)
    assert result["effective_route"] == "refuse"
    assert result["refusal_reason"] == "out_of_scope"


def test_intent_vague_routes_to_clarify_with_question():
    llm = _StubRouteLLM(
        [
            _ok(
                _parsed(
                    intent="vague",
                    companies=[],
                    years=[],
                    route="clarify",
                    clarification="Which company do you mean?",
                )
            )
        ]
    )
    result = _run(llm)
    assert result["effective_route"] == "clarify"
    assert result["clarification"] == "Which company do you mean?"


def test_unconfident_mention_routes_to_clarify_with_candidates():
    llm = _StubRouteLLM(
        [
            _ok(
                _parsed(
                    companies=[
                        CompanyMention(mentioned="the company", canonical=None, confident=False)
                    ],
                    route="clarify",
                    clarification=None,
                )
            )
        ]
    )
    result = _run(llm)
    assert result["effective_route"] == "clarify"
    assert "the company" in result["clarification"]


# --- history-aware resolution (found via live multi-turn testing; see
# docs/implementation-plan.md v2.6) ---


class _CapturingRouteLLM:
    def __init__(self, decision):
        self._decision = decision
        self.calls = 0
        self.last_messages = None

    def invoke(self, messages, config=None):
        self.calls += 1
        self.last_messages = messages
        return _ok(self._decision)


def test_history_is_threaded_before_the_current_question():
    llm = _CapturingRouteLLM(_parsed())
    node = build_route_node(COVERAGE, llm)
    node(
        {
            "question": "the revenue",
            "history": [("user", "suggest metrics for AMZN"), ("assistant", "Revenue or net income?")],
        },
        {},
    )
    roles_and_texts = llm.last_messages
    assert roles_and_texts[0][0] == "system"
    assert roles_and_texts[1] == ("human", "suggest metrics for AMZN")
    assert roles_and_texts[2] == ("ai", "Revenue or net income?")
    assert roles_and_texts[-1] == ("human", "the revenue")


def test_history_is_capped_to_history_max_messages():
    llm = _CapturingRouteLLM(_parsed())
    node = build_route_node(COVERAGE, llm, history_max_messages=2)
    long_history = [("user", "1"), ("assistant", "2"), ("user", "3"), ("assistant", "4")]
    node({"question": "q", "history": long_history}, {})
    # system + last 2 history entries + current question == 4 messages
    assert len(llm.last_messages) == 4
    assert llm.last_messages[1] == ("human", "3")
    assert llm.last_messages[2] == ("ai", "4")


def test_no_history_key_behaves_like_before():
    llm = _CapturingRouteLLM(_parsed())
    node = build_route_node(COVERAGE, llm)
    node({"question": "irrelevant"}, {})
    assert llm.last_messages == [("system", SYSTEM_PROMPT), ("human", "irrelevant")]
