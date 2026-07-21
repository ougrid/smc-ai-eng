"""AI SDK UI message stream v1 emitter, driven by LangGraph's
`astream_events()` (see docs/technical-execution-plan.md E6/E7).

`stream_agent_chat` is a pure function over an injected async iterator of
already-shaped LangChain/LangGraph events (`{event, name, tags, data}`) --
tests drive it with a scripted fake event list instead of a live graph
(see tests/test_sse.py); `run_graph_events` is the thin real-world adapter
that calls `graph.astream_events(...)`.

Stream-veto: `verify` runs after every `synthesize` attempt. `data-verify`
always reconciles the same id ("verify-1") across attempts. On a failed
verify, the NEXT text part (a retry's redraft, or the final refusal text)
gets a fresh id (`draft-2`, `draft-3`, ...) -- the frontend rule is "render
only the last text part", so the client never needs to know which attempt
succeeded, only which one is last.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

_ANSWER_KEY = '"answer"'
_SIMPLE_ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
}


class AnswerFieldExtractor:
    """Feed raw JSON-fragment deltas from the `synthesize` node's token
    stream; yields decoded characters of the `answer` field's value as soon
    as they're unambiguous. Schema field order (`reasoning` first,
    `citations` last) guarantees `answer` is the first string field
    encountered, and that nothing after its closing quote matters here --
    `citations` is delivered via a data part instead, never streamed.
    """

    def __init__(self) -> None:
        self._buffer = ""
        # One-way progression: key -> colon -> quote -> value -> done. Each
        # stage only consumes its buffer once its condition is met, so
        # partial progress (e.g. colon found but not yet the opening quote)
        # survives across arbitrarily small chunk boundaries.
        self._state = "key"

    def feed(self, delta: str) -> str:
        if self._state == "done" or not delta:
            return ""
        self._buffer += delta

        if self._state == "key":
            idx = self._buffer.find(_ANSWER_KEY)
            if idx == -1:
                # Keep a short tail in case the key straddles this chunk
                # and the next one.
                self._buffer = self._buffer[-(len(_ANSWER_KEY) - 1) :]
                return ""
            self._buffer = self._buffer[idx + len(_ANSWER_KEY) :]
            self._state = "colon"

        if self._state == "colon":
            colon = self._buffer.find(":")
            if colon == -1:
                return ""
            self._buffer = self._buffer[colon + 1 :]
            self._state = "quote"

        if self._state == "quote":
            quote = self._buffer.find('"')
            if quote == -1:
                return ""
            self._buffer = self._buffer[quote + 1 :]
            self._state = "value"

        out: list[str] = []
        i, n = 0, len(self._buffer)
        while i < n:
            ch = self._buffer[i]
            if ch == "\\":
                if i + 1 >= n:
                    break  # escape split across chunks -- wait for more
                nxt = self._buffer[i + 1]
                if nxt == "u":
                    if i + 6 > n:
                        break  # \uXXXX split across chunks
                    out.append(chr(int(self._buffer[i + 2 : i + 6], 16)))
                    i += 6
                    continue
                out.append(_SIMPLE_ESCAPES.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                self._state = "done"
                i += 1
                break
            out.append(ch)
            i += 1

        self._buffer = self._buffer[i:]
        return "".join(out)


def sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_agent_chat(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    message_id = str(uuid.uuid4())
    yield sse({"type": "start", "messageId": message_id})
    yield sse({"type": "start-step"})

    extractor = AnswerFieldExtractor()
    draft_index = 1
    text_id = f"draft-{draft_index}"
    text_open = False
    final_state: dict[str, Any] = {}

    async for event in events:
        kind = event.get("event")
        name = event.get("name")
        data = event.get("data", {})

        if kind == "on_chat_model_stream" and "synthesize" in event.get("tags", []):
            chunk = data.get("chunk")
            delta_text = getattr(chunk, "content", "") or ""
            piece = extractor.feed(delta_text)
            if piece:
                if not text_open:
                    yield sse({"type": "text-start", "id": text_id})
                    text_open = True
                yield sse({"type": "text-delta", "id": text_id, "delta": piece})
            continue

        if kind != "on_chain_end":
            continue

        output = data.get("output") or {}
        if not isinstance(output, dict):
            continue
        final_state.update(output)

        if name == "route":
            yield sse(
                {
                    "type": "data-route",
                    "id": "route-1",
                    "data": {
                        "route": output.get("effective_route"),
                        "companies": output.get("companies", []),
                        "years": output.get("years", []),
                    },
                }
            )
            if output.get("coverage_notes"):
                yield sse(
                    {
                        "type": "data-coverage",
                        "id": "coverage-1",
                        "data": {"notes": output["coverage_notes"]},
                    }
                )

        elif name in ("sql_retrieve", "vector_retrieve"):
            yield sse(
                {
                    "type": "data-citations",
                    "id": "citations-1",
                    "data": {
                        "sql_rows": final_state.get("sql_rows", []),
                        "chunks": final_state.get("chunks", []),
                    },
                }
            )

        elif name in ("refuse", "clarify"):
            answer = output.get("final_answer", "")
            yield sse({"type": "text-start", "id": text_id})
            if answer:
                yield sse({"type": "text-delta", "id": text_id, "delta": answer})
            yield sse({"type": "text-end", "id": text_id})
            text_open = False

        elif name == "synthesize":
            if not text_open:
                # The extractor emitted nothing during streaming (e.g. a
                # non-streaming stub, or the whole answer arrived in one
                # chunk before any partial match) -- fall back to emitting
                # the full answer in one shot.
                answer = output.get("final_answer", "")
                yield sse({"type": "text-start", "id": text_id})
                if answer:
                    yield sse({"type": "text-delta", "id": text_id, "delta": answer})
            yield sse({"type": "text-end", "id": text_id})
            text_open = False

        elif name == "verify":
            verify = output.get("verify") or {}
            yield sse({"type": "data-verify", "id": "verify-1", "data": verify})
            if not verify.get("ok", True):
                # Whatever text part comes next -- a redrafted retry, or the
                # final refusal template -- is a fresh part; the extractor
                # is per-synthesize-call state and must not carry over.
                draft_index += 1
                text_id = f"draft-{draft_index}"
                extractor = AnswerFieldExtractor()

    yield sse({"type": "finish-step"})
    # Everything below is already in state for free -- per docs/technical-
    # execution-plan.md E7, `debug` also carries the emitted SQL, computed
    # growth figures, and rejected vector chunks (incl. below-floor/
    # boilerplate/duplicate scores), not just the route node's own debug.
    debug = dict(final_state.get("debug", {}))
    if "verify" in final_state:
        debug["verify"] = final_state["verify"]
    if "sql" in final_state:
        debug["sql"] = final_state["sql"]
    if "computed" in final_state:
        debug["computed"] = final_state["computed"]
    if "rejected_chunks" in final_state:
        debug["rejected_chunks"] = final_state["rejected_chunks"]
    yield sse(
        {
            "type": "finish",
            "messageMetadata": {
                "route": final_state.get("effective_route"),
                "coverage_notes": final_state.get("coverage_notes", []),
                "citations": final_state.get("chunks", []) + final_state.get("sql_rows", []),
                "verify": final_state.get("verify"),
                "debug": debug,
            },
        }
    )
    yield "data: [DONE]\n\n"


async def run_graph_events(graph, question: str, history: list[tuple[str, str]]):
    async for event in graph.astream_events(
        {"question": question, "history": history}, version="v2"
    ):
        yield event
