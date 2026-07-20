"""app/chat/schemas.py had zero test coverage before this -- covers
`_text_of`'s last-text-part-only fix (the bug found via live multi-turn
testing: a message that survived a stream-veto retry has two "text"-type
parts, and joining both into history would resurface a discarded
fabricated draft alongside its correction), plus `to_history` and
`latest_user_text`.
"""

import pytest

from app.chat.schemas import ChatRequest, UIMessageIn, UIPartIn, latest_user_text, to_history


def _message(id_, role, *texts):
    return UIMessageIn(id=id_, role=role, parts=[UIPartIn(type="text", text=t) for t in texts])


def test_text_of_single_part_message_via_latest_user_text():
    req = ChatRequest(messages=[_message("u1", "user", "hello")])
    assert latest_user_text(req) == "hello"


def test_text_of_uses_only_the_last_part_not_a_join():
    # Simulates an assistant message that failed verify once (fabricated
    # draft-1) then was corrected (draft-2) -- both are separate "text"
    # parts on the same message by the time it's replayed as history.
    veto_message = _message("a1", "assistant", "fabricated $999,999M", "corrected $93,736M")
    req = ChatRequest(messages=[veto_message, _message("u2", "user", "and then?")])
    history = to_history(req)
    assert history == [("assistant", "corrected $93,736M")]


def test_text_of_empty_parts_list_is_empty_string():
    req = ChatRequest(messages=[UIMessageIn(id="a1", role="assistant", parts=[])])
    assert to_history(ChatRequest(messages=[req.messages[0], _message("u2", "user", "q")])) == [
        ("assistant", "")
    ]


def test_to_history_excludes_latest_message():
    req = ChatRequest(
        messages=[
            _message("u1", "user", "first"),
            _message("a1", "assistant", "reply"),
            _message("u2", "user", "latest"),
        ]
    )
    assert to_history(req) == [("user", "first"), ("assistant", "reply")]


def test_to_history_empty_messages_returns_empty():
    assert to_history(ChatRequest(messages=[])) == []


def test_latest_user_text_raises_on_empty_text():
    req = ChatRequest(messages=[_message("u1", "user", "")])
    with pytest.raises(ValueError):
        latest_user_text(req)


def test_latest_user_text_raises_when_no_user_message():
    req = ChatRequest(messages=[_message("a1", "assistant", "hi")])
    with pytest.raises(ValueError):
        latest_user_text(req)


def test_latest_user_text_finds_most_recent_user_message():
    req = ChatRequest(
        messages=[
            _message("u1", "user", "old question"),
            _message("a1", "assistant", "old answer"),
            _message("u2", "user", "new question"),
        ]
    )
    assert latest_user_text(req) == "new question"
