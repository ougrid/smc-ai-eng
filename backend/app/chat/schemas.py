"""Request parsing for the AI SDK `DefaultChatTransport` POST body.

The frontend's useChat posts `{id, messages: UIMessage[], trigger,
messageId}`; we only need `messages`, so everything else is ignored rather
than rejected -- the frontend's request shape can evolve without breaking us.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class UIPartIn(BaseModel):
    type: str
    text: str | None = None
    model_config = ConfigDict(extra="allow")  # data-*/file parts pass through untouched


class UIMessageIn(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    parts: list[UIPartIn]
    model_config = ConfigDict(extra="ignore")


class ChatRequest(BaseModel):
    id: str | None = None
    messages: list[UIMessageIn]
    model_config = ConfigDict(extra="ignore")  # tolerate trigger/messageId/future fields


def _text_of(message: UIMessageIn) -> str:
    """The message's answer text -- only the LAST "text"-type part, not a
    join of all of them. A message that survived a stream-veto retry
    (agent/sse.py) has more than one text part: the discarded fabricated
    draft, then its correction, each under a fresh part id. Joining every
    part would resurface the fabricated draft's text every time this
    message is later replayed as conversation history."""
    text_parts = [part.text or "" for part in message.parts if part.type == "text"]
    return text_parts[-1] if text_parts else ""


def latest_user_text(req: ChatRequest) -> str:
    """Text of the most recent user message. Raises ValueError if there is
    none or it's empty -- the router maps that to a 422."""
    for message in reversed(req.messages):
        if message.role == "user":
            text = _text_of(message)
            if not text:
                raise ValueError("latest user message has no text")
            return text
    raise ValueError("no user message found")


def to_history(req: ChatRequest) -> list[tuple[str, str]]:
    """(role, text) pairs for every message except the latest one."""
    if not req.messages:
        return []
    return [(m.role, _text_of(m)) for m in req.messages[:-1]]
