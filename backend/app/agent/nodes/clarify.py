"""Clarify node: terminal, no LLM call and no retrieval.

The route node already produced the clarifying question
(`state["clarification"]`, covering both vague intent and an unconfident
company mention); this node's only job is to surface it as the turn's
final answer. The user's reply is simply the next turn, arriving with full
conversation history, so no session state needs to be held anywhere. The
graph skips `verify` after this node -- there are no factual claims to
check.
"""

from app.agent.state import AgentState

_FALLBACK = "Could you clarify your question?"


def clarify_node(state: AgentState) -> dict:
    return {"final_answer": state.get("clarification") or _FALLBACK}
