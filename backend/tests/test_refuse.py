from app.agent.nodes.refuse import refuse_node


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
