"""verify node tests -- pure/offline, no LLM or graph involved (see
tests/test_synthesize.py for the equivalent pattern on the node before
this one in the pipeline).
"""

from app.agent.nodes.verify import build_verify_node


def _run(**state):
    return build_verify_node()(state)


def test_grounded_sql_number_passes():
    result = _run(
        final_answer="Apple's net income was $93,736M in 2024.",
        sql_rows=[{"company": "Apple", "year": 2024, "net_income": 93736000000}],
    )
    assert result["verify"]["ok"] is True
    assert result["verify"]["ungrounded"] == []
    assert result["verify"]["attempt"] == 1
    assert result["verify_attempts"] == 1
    assert "refusal_reason" not in result


def test_computed_growth_percent_passes_within_rounding_tolerance():
    result = _run(
        final_answer="Meta's revenue grew 22.2%, the highest of the four.",
        computed={"Meta": {"revenue_growth_2024_2025": 22.17}},
    )
    assert result["verify"]["ok"] is True


def test_fabricated_number_fails():
    result = _run(
        final_answer="Apple's net income was $999,999M in 2024.",
        sql_rows=[{"company": "Apple", "year": 2024, "net_income": 93736000000}],
    )
    assert result["verify"]["ok"] is False
    assert "999,999" in result["verify"]["ungrounded"][0].replace("$", "")
    assert result["refusal_reason"] == "unverified_numbers"


def test_years_in_state_are_always_grounded():
    result = _run(final_answer="In 2024, no SQL data was needed.", years=[2024])
    assert result["verify"]["ok"] is True


def test_page_citation_numbers_are_not_flagged():
    result = _run(final_answer="Meta grew due to ad revenue [Meta_10K.pdf, p.12].")
    assert result["verify"]["ok"] is True


def test_number_literally_quoted_in_chunk_text_passes():
    result = _run(
        final_answer="Google Cloud revenue reached $134.9 billion.",
        chunks=[{"company": "Google", "text": "Google Cloud revenue reached $134.9 billion in 2025."}],
    )
    assert result["verify"]["ok"] is True


def test_second_attempt_increments_counter():
    result = _run(
        final_answer="Apple's net income was $93,736M in 2024.",
        sql_rows=[{"company": "Apple", "year": 2024, "net_income": 93736000000}],
        verify_attempts=1,
    )
    assert result["verify"]["attempt"] == 2
    assert result["verify_attempts"] == 2


def test_no_numbers_in_answer_trivially_passes():
    result = _run(final_answer="I can't ground this qualitative claim for Microsoft.")
    assert result["verify"]["ok"] is True
    assert result["verify"]["ungrounded"] == []


def test_fabricated_large_number_does_not_collapse_to_a_degenerate_match():
    # A wide scale ladder (e.g. dividing by 1e6/1e9 like the SQL pool does)
    # can collapse a large fabricated number down near 0 or 1, producing a
    # degenerate "1"/"0" candidate that trivially substring-matches almost
    # any prose -- this must still be caught as ungrounded.
    result = _run(
        final_answer="Meta's revenue reached $999,999 billion.",
        chunks=[{"company": "Meta", "text": "Meta's revenue reached $134.9 billion in 2025."}],
    )
    assert result["verify"]["ok"] is False
    assert "999,999" in result["verify"]["ungrounded"][0]


def test_billions_paraphrase_of_a_millions_scale_chunk_figure_passes():
    # 10-K financial statements are conventionally "in millions" ("196,600"),
    # but narrative synthesis often rounds to billions ("$196.6 billion") --
    # both refer to the same $196.6B figure and must both ground it (found
    # via live testing once hybrid retrieval started surfacing more
    # numeric-bearing chunks, see agent/hybrid_tool.py).
    result = _run(
        final_answer="Meta's revenue was $196.6 billion.",
        chunks=[{"company": "Meta", "text": "Meta's total revenue was $196,600 for the year."}],
    )
    assert result["verify"]["ok"] is True
