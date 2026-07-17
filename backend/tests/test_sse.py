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

    assert kinds == ["start", "start-step", "data-route", "text-start", "text-delta", "text-end", "finish-step", "finish"]
    assert not any(p["type"] == "data-citations" for p in parsed)
    assert not any(p["type"] == "data-verify" for p in parsed)

    data_route = next(p for p in parsed if p["type"] == "data-route")
    assert data_route["data"]["route"] == "clarify"

    text_delta = next(p for p in parsed if p["type"] == "text-delta")
    assert text_delta["delta"] == "Which company do you mean?"


def test_sse_helper_keeps_non_ascii_readable():
    line = sse({"type": "text-delta", "delta": "กำไรสุทธิ"})
    assert "กำไรสุทธิ" in line
    assert "\\u" not in line
