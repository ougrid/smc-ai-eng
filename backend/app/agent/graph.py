"""build_graph wires the LangGraph agent: route -> {refuse, clarify,
sql_retrieve, vector_retrieve} -> synthesize -> verify -> {END, synthesize
(retry), refuse}. Every client (LLMs, tools, coverage map) is injected --
no module singletons -- so a graph can be built entirely from stubs in
tests.

verify is deterministic and bounded: one retry back into synthesize on a
failed numeric-consistency check, then a fail-closed refusal on the
second failure -- never a third attempt, never an annotated-but-unfixed
answer (docs/technical-execution-plan.md E6).
"""

from langgraph.graph import END, StateGraph

from app.agent.coverage import CoverageMap
from app.agent.nodes.clarify import clarify_node
from app.agent.nodes.refuse import refuse_node
from app.agent.nodes.route import RouteLLM, build_route_node
from app.agent.nodes.sql_retrieve import SqlLLM, build_sql_retrieve_node
from app.agent.nodes.synthesize import SynthesisLLM, build_synthesize_node
from app.agent.nodes.vector_retrieve import build_vector_retrieve_node
from app.agent.nodes.verify import build_verify_node
from app.agent.sql_tool import SqlTool
from app.agent.state import AgentState
from app.agent.vector_tool import VectorQueryable

MAX_VERIFY_ATTEMPTS = 2


def _after_route(state: AgentState) -> str:
    route = state.get("effective_route")
    if route in ("refuse", "clarify"):
        return route
    return "vector" if route == "vector" else "sql"  # "sql"/"both" -> SQL half first


def _after_sql(state: AgentState) -> str:
    if state.get("effective_route") == "both" and state.get("vector_companies"):
        return "vector"
    return "synthesize"


def _after_verify(state: AgentState) -> str:
    verify = state.get("verify") or {}
    if verify.get("ok"):
        return "ok"
    if state.get("verify_attempts", 0) >= MAX_VERIFY_ATTEMPTS:
        return "fail"
    return "retry"


def build_graph(
    *,
    coverage: CoverageMap,
    route_llm: RouteLLM,
    sql_llm: SqlLLM,
    synth_llm: SynthesisLLM,
    sql_tool: SqlTool,
    vector_tool: VectorQueryable,
    history_max_messages: int = 8,
):
    g = StateGraph(AgentState)

    g.add_node("route", build_route_node(coverage, route_llm, history_max_messages=history_max_messages))
    g.add_node("clarify", clarify_node)
    g.add_node("refuse", refuse_node)
    g.add_node("sql_retrieve", build_sql_retrieve_node(sql_llm, sql_tool))
    g.add_node("vector_retrieve", build_vector_retrieve_node(vector_tool))
    g.add_node(
        "synthesize", build_synthesize_node(synth_llm, history_max_messages=history_max_messages)
    )
    g.add_node("verify", build_verify_node())

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
    g.add_edge("synthesize", "verify")
    g.add_conditional_edges(
        "verify", _after_verify, {"ok": END, "retry": "synthesize", "fail": "refuse"}
    )
    g.add_edge("refuse", END)
    g.add_edge("clarify", END)

    return g.compile()
