"""LangGraph state shared across every node in the agent graph.

`total=False` because no single node touches every key -- LangGraph merges
each node's partial dict return into this state.
"""

from typing import Any, TypedDict


class VerifyResult(TypedDict, total=False):
    ok: bool
    ungrounded: list[str]
    attempt: int


class AgentState(TypedDict, total=False):
    # input
    question: str
    history: list[tuple[str, str]]

    # route node output (route node = LLM call + coverage gate together)
    route: dict[str, Any]  # RouteDecision.model_dump()
    effective_route: str  # "sql" | "vector" | "both" | "refuse" | "clarify"
    companies: list[str]  # canonical, gate-resolved
    vector_companies: list[str]  # subset of companies with an indexed 10-K
    years: list[int]
    metrics: list[str]  # router-inferred, English (revenue/net_income/...) -- used to reformulate vector queries
    coverage_notes: list[str]
    refusal_reason: str | None
    clarification: str | None

    # retrieval
    sql: str
    sql_rows: list[dict[str, Any]]
    computed: dict[str, Any]  # Python-calculated growth % -- part of verify's allowed set
    chunks: list[dict[str, Any]]
    rejected_chunks: list[dict[str, Any]]

    # synthesis + verification
    envelope: dict[str, Any]  # SynthesisEnvelope.model_dump()
    verify: VerifyResult
    verify_attempts: int
    final_answer: str

    # observability -- mirrored into the SSE `finish` message metadata
    debug: dict[str, Any]
