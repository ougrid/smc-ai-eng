"""Graph-wiring integration tests -- a compiled real LangGraph run end to
end over stubbed LLMs/tools (same stubbing pattern as the per-node tests),
covering the retry-then-refuse verify loop that only exists once every
node is wired together (see agent-output/day3-smoke-test-findings.md and
docs/technical-execution-plan.md E6 for the loop this exercises).
"""

from app.agent.coverage import CoverageMap
from app.agent.graph import build_graph
from app.agent.schemas import CompanyMention, RouteDecision, SynthesisEnvelope
from app.agent.vector_tool import Chunk, VectorResult

COVERAGE = CoverageMap(sql_years={"Meta": [2024, 2025]})


def _route_decision(**overrides):
    defaults = dict(
        reasoning="r",
        intent="financial",
        companies=[CompanyMention(mentioned="Meta", canonical="Meta", confident=True)],
        years=[2025],
        metrics=["revenue"],
        route="vector",
        clarification=None,
        language="en",
    )
    defaults.update(overrides)
    return RouteDecision(**defaults)


class _StubRouteLLM:
    def invoke(self, messages, config=None):
        return {"raw": None, "parsed": _route_decision(), "parsing_error": None}


class _StubVectorTool:
    def query(self, question, companies):
        return VectorResult(
            chunks=[
                Chunk(
                    id="c1",
                    company="Meta",
                    source="Meta_10K.pdf",
                    page=5,
                    text="Meta's revenue reached $134.9 billion in 2025.",
                    score=0.8,
                )
            ]
        )


class _StubSynthLLM:
    """Each entry is one invoke() response, in call order."""

    def __init__(self, envelopes):
        self._envelopes = list(envelopes)
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        return {"raw": None, "parsed": self._envelopes.pop(0), "parsing_error": None}


def _envelope(answer):
    return SynthesisEnvelope(reasoning="r", answer=answer, citations=[])


def _build(synth_llm):
    return build_graph(
        coverage=COVERAGE,
        route_llm=_StubRouteLLM(),
        sql_llm=None,
        synth_llm=synth_llm,
        sql_tool=None,
        vector_tool=_StubVectorTool(),
    )


def test_grounded_answer_passes_verify_on_first_attempt():
    graph = _build(_StubSynthLLM([_envelope("Meta's revenue reached $134.9 billion.")]))
    result = graph.invoke({"question": "why did Meta grow?", "history": []})
    assert result["verify"]["ok"] is True
    assert result["verify_attempts"] == 1
    assert result["final_answer"] == "Meta's revenue reached $134.9 billion."


def test_fabricated_number_retries_then_succeeds_on_corrected_redraft():
    synth_llm = _StubSynthLLM(
        [
            _envelope("Meta's revenue reached $999,999 billion."),  # fabricated
            _envelope("Meta's revenue reached $134.9 billion."),  # corrected, grounded
        ]
    )
    graph = _build(synth_llm)
    result = graph.invoke({"question": "why did Meta grow?", "history": []})
    assert synth_llm.calls == 2
    assert result["verify"]["ok"] is True
    assert result["verify_attempts"] == 2
    assert result["final_answer"] == "Meta's revenue reached $134.9 billion."


def test_persistently_fabricated_number_fails_closed_to_refusal():
    synth_llm = _StubSynthLLM(
        [
            _envelope("Meta's revenue reached $999,999 billion."),
            _envelope("Meta's revenue reached $888,888 billion."),  # still fabricated
        ]
    )
    graph = _build(synth_llm)
    result = graph.invoke({"question": "why did Meta grow?", "history": []})
    assert synth_llm.calls == 2
    assert result["verify"]["ok"] is False
    assert result["verify_attempts"] == 2
    assert result["refusal_reason"] == "unverified_numbers"
    assert "couldn't verify" in result["final_answer"] or "verify" in result["final_answer"].lower()
