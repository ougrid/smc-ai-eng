"""Day 2: a hardcoded stub that speaks the real wire protocol.

This retires the streaming-protocol risk (SSE framing, the AI SDK UI
message stream v1 shape, auth-guarding a streaming endpoint) before any
LLM/LangGraph code exists. Day 3 replaces `stream_stub_chat` with the real
graph-driven emitter; the endpoint signature and headers do not change.
"""

import json
import uuid
from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.auth.deps import get_current_user
from app.auth.models import User
from app.chat.schemas import ChatRequest, latest_user_text

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_stub_chat(question: str) -> Iterator[str]:
    message_id = str(uuid.uuid4())
    yield _sse({"type": "start", "messageId": message_id})
    yield _sse({"type": "start-step"})
    yield _sse(
        {
            "type": "data-route",
            "id": "route-1",
            "data": {"route": "stub", "companies": [], "years": [], "language": "auto"},
        }
    )

    text_id = "draft-1"
    yield _sse({"type": "text-start", "id": text_id})
    for chunk in (
        "This is a stub echo response -- ",
        f"you asked: {question!r}. ",
        "The real LangGraph agent lands in Day 3.",
    ):
        yield _sse({"type": "text-delta", "id": text_id, "delta": chunk})
    yield _sse({"type": "text-end", "id": text_id})

    yield _sse({"type": "finish-step"})
    yield _sse(
        {
            "type": "finish",
            "messageMetadata": {"route": "stub", "coverage_notes": [], "citations": []},
        }
    )
    yield "data: [DONE]\n\n"


@router.post("/chat")
def chat(body: ChatRequest, user: User = Depends(get_current_user)) -> StreamingResponse:
    try:
        question = latest_user_text(body)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No user message text found")
    return StreamingResponse(
        stream_stub_chat(question), media_type="text/event-stream", headers=SSE_HEADERS
    )
