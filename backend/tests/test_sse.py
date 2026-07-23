"""AnswerFieldExtractor unit tests, plus emitter tests over a scripted fake
event stream (no real LangGraph/LLM involved -- see docs/technical-
execution-plan.md E10).
"""

import itertools
import json

import pytest

from app.agent.sse import AnswerFieldExtractor, sse, stream_agent_chat


# --- AnswerFieldExtractor ---


def _feed_all(extractor, chunks):
    return "".join(extractor.feed(c) for c in chunks)


def test_extracts_plain_answer_in_one_chunk():
    extractor = AnswerFieldExtractor()
    payload = json.dumps({"reasoning": "r", "answer": "hello world", "citations": []})
    assert _feed_all(extractor, [payload]) == "hello world"


def test_extracts_answer_with_escaped_characters():
    extractor = AnswerFieldExtractor()
    payload = json.dumps({"reasoning": "r", "answer": 'He said "hi"\nnew line', "citations": []})
    assert _feed_all(extractor, [payload]) == 'He said "hi"\nnew line'


def test_extracts_answer_with_unicode_escapes_thai():
    extractor = AnswerFieldExtractor()
    # ensure_ascii=True forces \uXXXX escapes for the Thai text
    payload = json.dumps(
        {"reasoning": "r", "answer": "กำไรสุทธิ 93,736 ล้านดอลลาร์", "citations": []},
        ensure_ascii=True,
    )
    assert '\\u' in payload  # sanity: the escapes we're testing are actually present
    assert _feed_all(extractor, [payload]) == "กำไรสุทธิ 93,736 ล้านดอลลาร์"


def test_extracts_answer_split_across_arbitrary_chunk_boundaries():
    extractor = AnswerFieldExtractor()
    payload = json.dumps({"reasoning": "some reasoning here", "answer": "split me up", "citations": []})
    # split at every single character -- the hardest possible case
    chunks = list(payload)
    assert _feed_all(extractor, chunks) == "split me up"


def test_ignores_citations_after_answer_closes():
    extractor = AnswerFieldExtractor()
    payload = json.dumps(
        {"reasoning": "r", "answer": "the answer", "citations": [{"kind": "sql"}]}
    )
    assert _feed_all(extractor, [payload]) == "the answer"


def test_empty_feed_returns_empty_string():
    extractor = AnswerFieldExtractor()
    assert extractor.feed("") == ""


def test_feed_with_no_answer_key_yet_returns_empty():
    extractor = AnswerFieldExtractor()
    assert extractor.feed('{"reasoning": "still thinking...') == ""


# --- emitter over a scripted fake event stream ---


class _Chunk:
    def __init__(self, content):
        self.content = content


async def _events(scripted):
    for event in scripted:
        yield event


def _chat_model_stream_events(text, tags=("synthesize",)):
    return [
        {"event": "on_chat_model_stream", "tags": list(tags), "data": {"chunk": _Chunk(piece)}}
        for piece in text
    ]


async def _collect(scripted):
    return [line async for line in stream_agent_chat(_events(scripted))]


def _types_of(lines):
    return [json.loads(line[len("data: ") :].rstrip("\n\n")) for line in lines if line != "data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_sql_happy_path_part_ordering():
    envelope_json = json.dumps(
        {"reasoning": "r", "answer": "Apple net income was $93,736M in 2024.", "citations": []}
    )
    scripted = [
        {
            "event": "on_chain_end",
            "name": "route",
            "tags": [],
            "data": {
                "output": {
                    "effective_route": "sql",
                    "companies": ["Apple"],
                    "years": [2024],
                    "coverage_notes": [],
                }
            },
        },
        {
            "event": "on_chain_end",
            "name": "sql_retrieve",
            "tags": [],
            "data": {
                "output": {
                    "sql_rows": [{"company": "Apple", "year": 2024, "net_income": 93736000000}],
                    "computed": {},
                }
            },
        },
        *_chat_model_stream_events(envelope_json),
        {
            "event": "on_chain_end",
            "name": "synthesize",
            "tags": [],
            "data": {
                "output": {
                    "final_answer": "Apple net income was $93,736M in 2024.",
                    "envelope": {"reasoning": "r", "answer": "...", "citations": []},
                }
            },
        },
    ]

    parsed = _types_of(await _collect(scripted))
    kinds = [p["type"] for p in parsed]

    # text-delta fires once per streamed character here (the fake event
    # stream feeds the extractor one char at a time) -- collapse consecutive
    # repeats to check part *ordering* without pinning an exact delta count.
    collapsed = [k for k, _ in itertools.groupby(kinds)]
    assert collapsed == [
        "start",
        "start-step",
        "data-route",
        "data-citations",
        "text-start",
        "text-delta",
        "text-end",
        "data-debug",
        "finish-step",
        "finish",
    ]

    deltas = "".join(p["delta"] for p in parsed if p["type"] == "text-delta")
    assert deltas == "Apple net income was $93,736M in 2024."

    finish = parsed[-1]
    assert finish["messageMetadata"]["route"] == "sql"


@pytest.mark.asyncio
async def test_clarify_sequence_has_no_citations_or_verify_part():
    scripted = [
        {
            "event": "on_chain_end",
            "name": "route",
            "tags": [],
            "data": {
                "output": {
                    "effective_route": "clarify",
                    "companies": [],
                    "years": [],
                    "coverage_notes": [],
                    "clarification": "Which company do you mean?",
                }
            },
        },
        {
            "event": "on_chain_end",
            "name": "clarify",
            "tags": [],
            "data": {"output": {"final_answer": "Which company do you mean?"}},
        },
    ]

    parsed = _types_of(await _collect(scripted))
    kinds = [p["type"] for p in parsed]

    assert kinds == ["start", "start-step", "data-route", "text-start", "text-delta", "text-end", "data-debug", "finish-step", "finish"]
    assert not any(p["type"] == "data-citations" for p in parsed)
    assert not any(p["type"] == "data-verify" for p in parsed)

    data_route = next(p for p in parsed if p["type"] == "data-route")
    assert data_route["data"]["route"] == "clarify"

    text_delta = next(p for p in parsed if p["type"] == "text-delta")
    assert text_delta["delta"] == "Which company do you mean?"


@pytest.mark.asyncio
async def test_capability_sequence_has_no_citations_or_verify_part():
    scripted = [
        {
            "event": "on_chain_end",
            "name": "route",
            "tags": [],
            "data": {
                "output": {
                    "effective_route": "capability",
                    "companies": [],
                    "years": [],
                    "coverage_notes": [],
                }
            },
        },
        {
            "event": "on_chain_end",
            "name": "capability",
            "tags": [],
            "data": {"output": {"final_answer": "I can help with financials for 49 companies..."}},
        },
    ]

    parsed = _types_of(await _collect(scripted))
    kinds = [p["type"] for p in parsed]

    assert kinds == ["start", "start-step", "data-route", "text-start", "text-delta", "text-end", "data-debug", "finish-step", "finish"]
    assert not any(p["type"] == "data-citations" for p in parsed)
    assert not any(p["type"] == "data-verify" for p in parsed)

    data_route = next(p for p in parsed if p["type"] == "data-route")
    assert data_route["data"]["route"] == "capability"

    text_delta = next(p for p in parsed if p["type"] == "text-delta")
    assert text_delta["delta"] == "I can help with financials for 49 companies..."


def _verify_event(ok, ungrounded, attempt):
    return {
        "event": "on_chain_end",
        "name": "verify",
        "tags": [],
        "data": {"output": {"verify": {"ok": ok, "ungrounded": ungrounded, "attempt": attempt}}},
    }


def _synthesize_end_event(final_answer):
    return {
        "event": "on_chain_end",
        "name": "synthesize",
        "tags": [],
        "data": {"output": {"final_answer": final_answer}},
    }


@pytest.mark.asyncio
async def test_stream_veto_retry_uses_a_fresh_draft_id_and_reconciles_verify_part():
    envelope_1 = json.dumps({"reasoning": "r", "answer": "fabricated $999,999M", "citations": []})
    envelope_2 = json.dumps({"reasoning": "r", "answer": "corrected $93,736M", "citations": []})
    scripted = [
        *_chat_model_stream_events(envelope_1),
        _synthesize_end_event("fabricated $999,999M"),
        _verify_event(False, ["999,999"], 1),
        *_chat_model_stream_events(envelope_2),
        _synthesize_end_event("corrected $93,736M"),
        _verify_event(True, [], 2),
    ]

    lines = await _collect(scripted)
    parsed = _types_of(lines)

    text_starts = [p for p in parsed if p["type"] == "text-start"]
    text_ends = [p for p in parsed if p["type"] == "text-end"]
    assert [p["id"] for p in text_starts] == ["draft-1", "draft-2"]
    assert [p["id"] for p in text_ends] == ["draft-1", "draft-2"]

    verify_parts = [p for p in parsed if p["type"] == "data-verify"]
    assert len(verify_parts) == 2
    assert all(p["id"] == "verify-1" for p in verify_parts)
    assert verify_parts[0]["data"]["ok"] is False
    assert verify_parts[1]["data"]["ok"] is True

    draft_2_deltas = "".join(
        p["delta"]
        for p in parsed
        if p["type"] == "text-delta"
        and parsed.index(p) > parsed.index(text_starts[1])
    )
    assert draft_2_deltas == "corrected $93,736M"

    finish = parsed[-1]
    assert finish["messageMetadata"]["verify"] == {"ok": True, "ungrounded": [], "attempt": 2}


@pytest.mark.asyncio
async def test_stream_veto_final_failure_streams_refusal_as_a_third_draft():
    envelope_1 = json.dumps({"reasoning": "r", "answer": "fabricated $999,999M", "citations": []})
    envelope_2 = json.dumps({"reasoning": "r", "answer": "still fabricated $888,888M", "citations": []})
    scripted = [
        *_chat_model_stream_events(envelope_1),
        _synthesize_end_event("fabricated $999,999M"),
        _verify_event(False, ["999,999"], 1),
        *_chat_model_stream_events(envelope_2),
        _synthesize_end_event("still fabricated $888,888M"),
        _verify_event(False, ["888,888"], 2),
        {
            "event": "on_chain_end",
            "name": "refuse",
            "tags": [],
            "data": {"output": {"final_answer": "I couldn't verify all the numbers."}},
        },
    ]

    parsed = _types_of(await _collect(scripted))
    text_starts = [p for p in parsed if p["type"] == "text-start"]
    assert [p["id"] for p in text_starts] == ["draft-1", "draft-2", "draft-3"]

    last_text = next(p for p in reversed(parsed) if p["type"] == "text-delta")
    assert last_text["delta"] == "I couldn't verify all the numbers."

    verify_parts = [p for p in parsed if p["type"] == "data-verify"]
    assert [p["data"]["ok"] for p in verify_parts] == [False, False]


def _chain_start_event(name):
    return {"event": "on_chain_start", "name": name, "tags": [], "data": {"input": {}}}


@pytest.mark.asyncio
async def test_data_status_parts_emitted_on_node_start_in_order_with_fixed_id():
    envelope_json = json.dumps({"reasoning": "r", "answer": "answer text", "citations": []})
    scripted = [
        {
            "event": "on_chain_end",
            "name": "route",
            "tags": [],
            "data": {"output": {"effective_route": "both", "companies": [], "years": [], "coverage_notes": []}},
        },
        # nodes not in the status map (the graph itself, routing) must not
        # emit a status part
        _chain_start_event("LangGraph"),
        _chain_start_event("route"),
        _chain_start_event("sql_retrieve"),
        {"event": "on_chain_end", "name": "sql_retrieve", "tags": [], "data": {"output": {"sql_rows": []}}},
        _chain_start_event("vector_retrieve"),
        {"event": "on_chain_end", "name": "vector_retrieve", "tags": [], "data": {"output": {"chunks": []}}},
        _chain_start_event("synthesize"),
        *_chat_model_stream_events(envelope_json),
        _synthesize_end_event("answer text"),
    ]

    parsed = _types_of(await _collect(scripted))
    status_parts = [p for p in parsed if p["type"] == "data-status"]

    # one per real work node, in node order, unknown chains ignored
    assert [p["data"]["stage"] for p in status_parts] == ["sql", "vector", "synthesize"]
    # fixed id so successive emissions reconcile/replace in the AI SDK
    assert all(p["id"] == "status-1" for p in status_parts)
    # every status carries a human-facing label
    assert all(p["data"].get("label") for p in status_parts)

    # the status labels precede the answer text (that's the whole point --
    # they fill the dead air before synthesis streams)
    first_status_idx = next(i for i, p in enumerate(parsed) if p["type"] == "data-status")
    first_text_idx = next(i for i, p in enumerate(parsed) if p["type"] == "text-start")
    assert first_status_idx < first_text_idx

    # stream is still well-formed AI SDK v1: framed by start/finish
    kinds = [p["type"] for p in parsed]
    assert kinds[0] == "start"
    assert kinds[1] == "start-step"
    assert kinds[-1] == "finish"


@pytest.mark.asyncio
async def test_synthesize_status_re_emitted_on_verify_retry_with_same_id():
    envelope_1 = json.dumps({"reasoning": "r", "answer": "fabricated $999,999M", "citations": []})
    envelope_2 = json.dumps({"reasoning": "r", "answer": "corrected $93,736M", "citations": []})
    scripted = [
        _chain_start_event("synthesize"),
        *_chat_model_stream_events(envelope_1),
        _synthesize_end_event("fabricated $999,999M"),
        _verify_event(False, ["999,999"], 1),
        # retry: synthesize starts again -> status re-emitted under same id
        _chain_start_event("synthesize"),
        *_chat_model_stream_events(envelope_2),
        _synthesize_end_event("corrected $93,736M"),
        _verify_event(True, [], 2),
    ]

    parsed = _types_of(await _collect(scripted))
    status_parts = [p for p in parsed if p["type"] == "data-status"]

    assert [p["data"]["stage"] for p in status_parts] == ["synthesize", "synthesize"]
    assert all(p["id"] == "status-1" for p in status_parts)


@pytest.mark.asyncio
async def test_data_debug_part_carries_full_pipeline_payload():
    envelope_json = json.dumps(
        {"reasoning": "r", "answer": "Apple net income was $93,736M in 2024.", "citations": []}
    )
    scripted = [
        {
            "event": "on_chain_end",
            "name": "route",
            "tags": [],
            "data": {
                "output": {
                    "effective_route": "sql",
                    "companies": ["Apple"],
                    "years": [2024],
                    "coverage_notes": [],
                    "debug": {
                        "route_decision": {"intent": "financial", "language": "en"},
                        "gate_result": {"companies": ["Apple"], "refusal_reason": None},
                    },
                }
            },
        },
        {
            "event": "on_chain_end",
            "name": "sql_retrieve",
            "tags": [],
            "data": {
                "output": {
                    "sql_rows": [{"company": "Apple", "year": 2024, "net_income": 93736000000}],
                    "sql": "SELECT net_income FROM financial_data WHERE company = 'Apple'",
                    "computed": {"Apple": {"net_income_2024": 93736000000}},
                }
            },
        },
        *_chat_model_stream_events(envelope_json),
        _synthesize_end_event("Apple net income was $93,736M in 2024."),
        _verify_event(True, [], 1),
    ]

    parsed = _types_of(await _collect(scripted))
    debug_parts = [p for p in parsed if p["type"] == "data-debug"]

    # exactly one debug part, fixed id, emitted before finish-step
    assert len(debug_parts) == 1
    assert debug_parts[0]["id"] == "debug-1"
    kinds = [p["type"] for p in parsed]
    assert kinds.index("data-debug") < kinds.index("finish-step")

    data = debug_parts[0]["data"]
    # carries the router's own debug, the executed SQL, the Python-computed
    # figures, and the deterministic verify result -- the demo/dev payload
    assert data["route_decision"]["intent"] == "financial"
    assert data["gate_result"]["refusal_reason"] is None
    assert data["sql"].startswith("SELECT")
    assert data["computed"]["Apple"]["net_income_2024"] == 93736000000
    assert data["verify"] == {"ok": True, "ungrounded": [], "attempt": 1}


def test_sse_helper_keeps_non_ascii_readable():
    line = sse({"type": "text-delta", "delta": "กำไรสุทธิ"})
    assert "กำไรสุทธิ" in line
    assert "\\u" not in line
