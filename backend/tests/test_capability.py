from app.agent.coverage import CoverageMap
from app.agent.nodes.capability import build_capability_node

COVERAGE = CoverageMap(
    sql_years={
        "Apple": [2022, 2023, 2024, 2025],
        "Google": [2022, 2023, 2024, 2025],
        "Meta": [2022, 2023, 2024, 2025],
        "Microsoft": [2022, 2023, 2024, 2025],
        "BlackRock": [2022, 2023],
    }
)


def test_capability_answer_names_real_companies_and_years():
    node = build_capability_node(COVERAGE)
    result = node({})
    answer = result["final_answer"]
    # Derived from the live coverage map, not hardcoded -- must mention the
    # actual 10-K-covered companies and the actual year range.
    assert "Apple" in answer and "Meta" in answer
    assert "2022" in answer and "2025" in answer
    assert str(len(COVERAGE.sql_years)) in answer


def test_capability_answer_never_mentions_uncovered_company():
    node = build_capability_node(COVERAGE)
    result = node({})
    assert "Netflix" not in result["final_answer"]


def test_capability_repeat_question_gets_a_varied_answer():
    node = build_capability_node(COVERAGE)
    first = node({})["final_answer"]
    second = node(
        {"history": [("user", "what can you do?"), ("assistant", first)]}
    )["final_answer"]
    assert first != second


def test_capability_answer_in_thai_when_language_is_thai():
    node = build_capability_node(COVERAGE)
    result = node({"route": {"language": "th"}})
    assert any("฀" <= ch <= "๿" for ch in result["final_answer"])
