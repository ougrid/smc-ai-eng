"""vector_retrieve: query the vector tool for whichever companies the
coverage gate marked vector-eligible (`state["vector_companies"]`) --
a subset of `state["companies"]` in a hybrid route where one company has
SQL data but no indexed 10-K (Microsoft in the Q3 baseline).
"""

from dataclasses import asdict
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agent.state import AgentState
from app.agent.vector_tool import VectorTool


def build_vector_retrieve_node(vector_tool: VectorTool):
    def _node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        result = vector_tool.query(state["question"], state.get("vector_companies", []))
        return {
            "chunks": [asdict(c) for c in result.chunks],
            "rejected_chunks": [asdict(c) for c in result.rejected],
        }

    return _node
