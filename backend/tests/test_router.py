"""Route node tests -- offline, stubbing the structured-output-bound LLM
runnable directly (the `RouteLLM` protocol) rather than a real chat model.
"""

from app.agent.coverage import CoverageMap
from app.agent.nodes.route import build_route_node
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
