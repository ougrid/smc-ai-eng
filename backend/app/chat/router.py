"""POST /api/chat -- the LangGraph agent's SSE stream.

The Day-2 hardcoded stub retired the streaming-protocol risk (SSE framing,
the AI SDK UI message stream v1 shape, auth-guarding a streaming endpoint)
before any LLM/LangGraph code existed. This now drives the real graph via
`app.agent.sse.stream_agent_chat`; the endpoint signature and headers are
unchanged from Day 2.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.agent.sse import run_graph_events, stream_agent_chat
from app.auth.deps import get_current_user
from app.auth.models import User
from app.chat.schemas import ChatRequest, latest_user_text, to_history

router = APIRouter(prefix="/api", tags=["chat"])

SSE_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("/chat")
def chat(
    body: ChatRequest, request: Request, user: User = Depends(get_current_user)
) -> StreamingResponse:
    try:
        question = latest_user_text(body)
    except ValueError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No user message text found")

    graph = request.app.state.graph
    events = run_graph_events(graph, question, to_history(body))
    return StreamingResponse(
        stream_agent_chat(events), media_type="text/event-stream", headers=SSE_HEADERS
    )
