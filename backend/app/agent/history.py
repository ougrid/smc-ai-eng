"""Conversation history helpers shared by `nodes/route.py` and
`nodes/synthesize.py`.

Deliberately a capped VERBATIM window, never summarized -- see
docs/implementation-plan.md's Future-improvements entry on hierarchical
summarization for why: this app's sessions are a handful of short
exchanges, so a small fixed window fully covers real follow-ups
("the revenue" after a company was named two turns ago) without paying
for, or risking the hallucination surface of, an LLM summarization call.
"""

from app.agent.state import AgentState

_ROLE_MAP = {"user": "human", "assistant": "ai"}


def trim_history(history: list[tuple[str, str]], max_messages: int) -> list[tuple[str, str]]:
    """Keep only the last `max_messages` entries -- oldest dropped first."""
    if max_messages <= 0:
        return []
    return history[-max_messages:]


def to_lc_messages(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Map `(role, text)` pairs from `AgentState["history"]` (frontend
    UIMessage roles) to LangChain tuple-message roles. Anything that isn't
    `user`/`assistant` (e.g. a stray `system`-role entry) is dropped -- it
    isn't a turn worth replaying, and shouldn't collide with the node's own
    system prompt."""
    return [(_ROLE_MAP[role], text) for role, text in history if role in _ROLE_MAP]


def history_messages(state: AgentState, max_messages: int) -> list[tuple[str, str]]:
    """Convenience wrapper: trim then map `state["history"]` in one call."""
    return to_lc_messages(trim_history(state.get("history") or [], max_messages))
