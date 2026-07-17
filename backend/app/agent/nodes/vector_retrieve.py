"""vector_retrieve: query the vector tool for whichever companies the
coverage gate marked vector-eligible (`state["vector_companies"]`) --
a subset of `state["companies"]` in a hybrid route where one company has
SQL data but no indexed 10-K (Microsoft in the Q3 baseline).

The embedding query is reformulated, not the raw user question verbatim:
the 10-K chunks are all English business prose, but questions are often
Thai (or a bare "why did revenue grow?"), so an English business-narrative
suffix -- built from the router's already-extracted `metrics` -- is
appended before embedding. This is what the Day-3 smoke test
(agent-output/day3-smoke-test-findings.md) identified as the fix for top
matches being generic product lists/legal text instead of strategy
narrative: it doesn't change what's indexed, only what's asked for.
"""

from dataclasses import asdict
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agent.state import AgentState
from app.agent.vector_tool import VectorTool

_NARRATIVE_HINT = "business strategy, revenue structure, growth drivers, competitive strengths"


def _reformulate(question: str, metrics: list[str]) -> str:
    if not metrics:
        return f"{question}\n\nRelevant context: {_NARRATIVE_HINT}."
    metric_labels = ", ".join(m.replace("_", " ") for m in metrics)
    return f"{question}\n\nRelevant context: {metric_labels}, {_NARRATIVE_HINT}."


def build_vector_retrieve_node(vector_tool: VectorTool):
    def _node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        query = _reformulate(state["question"], state.get("metrics", []))
        result = vector_tool.query(query, state.get("vector_companies", []))
        return {
            "chunks": [asdict(c) for c in result.chunks],
            "rejected_chunks": [asdict(c) for c in result.rejected],
        }

    return _node
