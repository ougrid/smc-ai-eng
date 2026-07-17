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
        chunks = [asdict(c) for c in result.chunks]

        # Deterministic gap detection: a company can be vector-eligible (has an
        # indexed 10-K) yet still end up with zero kept chunks for THIS question
        # (everything retrieved was below-floor or boilerplate). Left silent,
        # the Day-3 smoke test found the model just drops that company's
        # qualitative half rather than flagging it -- so surface it the same
        # way the existing "no 10-K at all" gap is surfaced: as a coverage
        # note the synthesis prompt is already instructed to reproduce verbatim.
        covered = {c["company"] for c in chunks}
        notes = list(state.get("coverage_notes", []))
        for company in state.get("vector_companies", []):
            if company not in covered:
                notes.append(
                    f"No substantive 10-K excerpts were retrieved for {company} to "
                    f"ground the qualitative portion of this question -- state "
                    f"explicitly that this cannot be grounded for {company}, "
                    "don't omit it."
                )

        return {
            "chunks": chunks,
            "rejected_chunks": [asdict(c) for c in result.rejected],
            "coverage_notes": notes,
        }

    return _node
