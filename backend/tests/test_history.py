from app.agent.history import history_messages, to_lc_messages, trim_history


def test_trim_history_keeps_last_n():
    history = [("user", "1"), ("assistant", "2"), ("user", "3"), ("assistant", "4")]
    assert trim_history(history, 2) == [("user", "3"), ("assistant", "4")]


def test_trim_history_noop_when_under_cap():
    history = [("user", "1"), ("assistant", "2")]
    assert trim_history(history, 8) == history


def test_trim_history_zero_or_negative_cap_drops_everything():
    history = [("user", "1")]
    assert trim_history(history, 0) == []
    assert trim_history(history, -1) == []


def test_to_lc_messages_maps_user_and_assistant():
    history = [("user", "hi"), ("assistant", "hello")]
    assert to_lc_messages(history) == [("human", "hi"), ("ai", "hello")]


def test_to_lc_messages_drops_unmapped_roles():
    history = [("user", "hi"), ("system", "ignored"), ("assistant", "hello")]
    assert to_lc_messages(history) == [("human", "hi"), ("ai", "hello")]


def test_history_messages_trims_then_maps():
    state = {
        "history": [
            ("user", "1"),
            ("assistant", "2"),
            ("user", "3"),
            ("assistant", "4"),
        ]
    }
    assert history_messages(state, 2) == [("human", "3"), ("ai", "4")]


def test_history_messages_handles_missing_history_key():
    assert history_messages({}, 8) == []
