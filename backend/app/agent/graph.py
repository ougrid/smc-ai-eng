"""build_graph wires the LangGraph agent: route -> {refuse, clarify,
sql_retrieve, vector_retrieve} -> synthesize -> END. Every client (LLMs,
tools, coverage map) is injected -- no module singletons -- so a graph can
be built entirely from stubs in tests.

Day-3 scope: no `verify` node yet -- synthesize goes straight to END. Day 4
inserts the numeric-consistency guard between them with a bounded
retry-then-refuse loop; this module's wiring will grow one more
conditional edge at that point, not a rewrite.
"""

from langgraph.graph import END, StateGraph

from app.agent.coverage import CoverageMap
from app.agent.nodes.clarify import clarify_node
from app.agent.nodes.refuse import refuse_node
from app.agent.nodes.route import RouteLLM, build_route_node
from app.agent.nodes.sql_retrieve import SqlLLM, build_sql_retrieve_node
from app.agent.nodes.synthesize import SynthesisLLM, build_synthesize_node
from app.agent.nodes.vector_retrieve import build_vector_retrieve_node
from app.agent.sql_tool import SqlTool
from app.agent.state import AgentState
from app.agent.vector_tool import VectorTool


def _after_route(state: AgentState) -> str:
    route = state.get("effective_route")
    if route in ("refuse", "clarify"):
        return route
    return "vector" if route == "vector" else "sql"  # "sql"/"both" -> SQL half first


def _after_sql(state: AgentState) -> str:
    if state.get("effective_route") == "both" and state.get("vector_companies"):
        return "vector"
    return "synthesize"


def build_graph(
    *,
    coverage: CoverageMap,
    route_llm: RouteLLM,
    sql_llm: SqlLLM,
    synth_llm: SynthesisLLM,
    sql_tool: SqlTool,
    vector_tool: VectorTool,
):
    g = StateGraph(AgentState)

    g.add_node("route", build_route_node(coverage, route_llm))
    g.add_node("clarify", clarify_node)
    g.add_node("refuse", refuse_node)
    g.add_node("sql_retrieve", build_sql_retrieve_node(sql_llm, sql_tool))
    g.add_node("vector_retrieve", build_vector_retrieve_node(vector_tool))
    g.add_node("synthesize", build_synthesize_node(synth_llm))

    g.set_entry_point("route")
    g.add_conditional_edges(
        "route",
        _after_route,
        {
            "refuse": "refuse",
            "clarify": "clarify",
            "sql": "sql_retrieve",
            "vector": "vector_retrieve",
        },
    )
    g.add_conditional_edges(
        "sql_retrieve", _after_sql, {"vector": "vector_retrieve", "synthesize": "synthesize"}
    )
    g.add_edge("vector_retrieve", "synthesize")
    g.add_edge("synthesize", END)
    g.add_edge("refuse", END)
    g.add_edge("clarify", END)

    return g.compile()
