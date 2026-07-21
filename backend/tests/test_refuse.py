from app.agent.nodes.refuse import (
    _DATA_UNAVAILABLE,
    _OUT_OF_SCOPE,
    _UNVERIFIED_CITATIONS,
    refuse_node,
)


def test_out_of_scope_english():
    result = refuse_node({"refusal_reason": "out_of_scope", "route": {"language": "en"}})
    assert "financials" in result["final_answer"]


def test_out_of_scope_thai():
    result = refuse_node({"refusal_reason": "out_of_scope", "route": {"language": "th"}})
    assert "ข้อมูลการเงิน" in result["final_answer"]


def test_data_unavailable_appends_coverage_notes():
    result = refuse_node(
        {
            "refusal_reason": "unknown_company",
            "route": {"language": "en"},
            "coverage_notes": ["No data available for: Siemens."],
        }
    )
    assert "don't have the data" in result["final_answer"]
    assert "Siemens" in result["final_answer"]


def test_defaults_to_english_when_language_missing():
    result = refuse_node({"refusal_reason": "unknown_company"})
    assert "don't have the data" in result["final_answer"]


def test_unverified_citations_english():
    result = refuse_node({"refusal_reason": "unverified_citations", "route": {"language": "en"}})
    assert "citations didn't match" in result["final_answer"]


def test_unverified_citations_thai():
    result = refuse_node({"refusal_reason": "unverified_citations", "route": {"language": "th"}})
    assert "การอ้างอิง" in result["final_answer"]


# --- anti-parroting: variant selection driven by prior-refusal count ---


def _history_with_refusals(*texts):
    """A conversation whose assistant turns are the given refusal strings."""
    history = []
    for text in texts:
        history.append(("user", "some out-of-scope question"))
        history.append(("assistant", text))
    return history


def test_first_refusal_uses_plain_variant_english():
    # No prior refusal in history -> variant 0, identical to the original text.
    result = refuse_node({"refusal_reason": "out_of_scope", "route": {"language": "en"}})
    assert result["final_answer"] == _OUT_OF_SCOPE["en"][0]


def test_repeat_out_of_scope_uses_acknowledging_variant_english():
    state = {
        "refusal_reason": "out_of_scope",
        "route": {"language": "en"},
        "history": _history_with_refusals(_OUT_OF_SCOPE["en"][0]),
    }
    result = refuse_node(state)
    assert result["final_answer"] == _OUT_OF_SCOPE["en"][1]
    assert result["final_answer"] != _OUT_OF_SCOPE["en"][0]
    assert "As I mentioned" in result["final_answer"]


def test_repeat_out_of_scope_uses_acknowledging_variant_thai():
    state = {
        "refusal_reason": "out_of_scope",
        "route": {"language": "th"},
        "history": _history_with_refusals(_OUT_OF_SCOPE["th"][0]),
    }
    result = refuse_node(state)
    assert result["final_answer"] == _OUT_OF_SCOPE["th"][1]
    assert result["final_answer"] != _OUT_OF_SCOPE["th"][0]
    assert "อย่างที่เรียนไป" in result["final_answer"]


def test_third_refusal_clamps_to_last_variant():
    # Two prior refusals -> index 2 (the most explicit steer); a further
    # refusal would clamp to the same last variant rather than overflow.
    state = {
        "refusal_reason": "out_of_scope",
        "route": {"language": "en"},
        "history": _history_with_refusals(_OUT_OF_SCOPE["en"][0], _OUT_OF_SCOPE["en"][1]),
    }
    result = refuse_node(state)
    assert result["final_answer"] == _OUT_OF_SCOPE["en"][2]


def test_repeat_data_unavailable_still_appends_coverage_notes():
    state = {
        "refusal_reason": "unknown_company",
        "route": {"language": "en"},
        "coverage_notes": ["No data available for: Siemens."],
        "history": _history_with_refusals(_DATA_UNAVAILABLE["en"][0]),
    }
    result = refuse_node(state)
    # Repeat variant is used, and the coverage note is still appended verbatim.
    assert result["final_answer"].startswith(_DATA_UNAVAILABLE["en"][1])
    assert "Siemens" in result["final_answer"]
    assert "still don't have data for that" in result["final_answer"]


def test_prior_refusal_counted_across_reasons_and_languages():
    # A prior refusal of a DIFFERENT reason/language still counts as a repeat.
    state = {
        "refusal_reason": "out_of_scope",
        "route": {"language": "en"},
        "history": _history_with_refusals(_UNVERIFIED_CITATIONS["th"][0]),
    }
    result = refuse_node(state)
    assert result["final_answer"] == _OUT_OF_SCOPE["en"][1]


def test_non_refusal_assistant_turns_do_not_count_as_repeats():
    # An ordinary grounded answer in history must not bump the variant index.
    state = {
        "refusal_reason": "out_of_scope",
        "route": {"language": "en"},
        "history": [
            ("user", "What was Apple's net income in 2024?"),
            ("assistant", "Apple's net income in 2024 was $93,736M."),
        ],
    }
    result = refuse_node(state)
    assert result["final_answer"] == _OUT_OF_SCOPE["en"][0]
