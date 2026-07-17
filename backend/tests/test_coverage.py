"""Coverage gate tests -- all offline, no LLM or live DB involved.

Each case in fixtures/routing_cases.json stubs a route-LLM output
(`route_decision`) and asserts what the deterministic gate does with it
against a seeded test engine. This is the fixture set `agent/nodes/route.py`
will reuse once the LLM call exists (see docs/technical-execution-plan.md
E10) -- the gate logic itself needs no LLM to test.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

from app.agent.coverage import apply_gate, build_coverage, resolve_company

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "routing_cases.json").read_text(encoding="utf-8")
)

# Mirrors the real financial_data years for the companies these fixtures touch.
_SEED_YEARS = {
    "Apple": [2022, 2023, 2024, 2025],
    "Amazon": [2022, 2023, 2024, 2025],
    "Google": [2022, 2023, 2024, 2025],
    "Meta": [2022, 2023, 2024, 2025],
    "Microsoft": [2022, 2023, 2024, 2025],
    "BlackRock": [2022, 2023],
    "Shopify": [2024, 2025],
}


@pytest.fixture()
def coverage():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE financial_data (company TEXT, year INTEGER)"))
        for company, years in _SEED_YEARS.items():
            for year in years:
                conn.execute(
                    text("INSERT INTO financial_data (company, year) VALUES (:c, :y)"),
                    {"c": company, "y": year},
                )
    return build_coverage(engine)


def test_build_coverage_years(coverage):
    assert coverage.sql_years["BlackRock"] == [2022, 2023]
    assert coverage.sql_years["Shopify"] == [2024, 2025]
    assert coverage.has_vector("Apple")
    assert not coverage.has_vector("BlackRock")


def _decision_from(payload: dict) -> SimpleNamespace:
    companies = [SimpleNamespace(**c) for c in payload["companies"]]
    return SimpleNamespace(**{**payload, "companies": companies})


@pytest.mark.parametrize("case", FIXTURES, ids=[c["id"] for c in FIXTURES])
def test_routing_cases(coverage, case):
    decision = _decision_from(case["route_decision"])
    expected = case["expected"]

    result = apply_gate(decision, coverage)

    assert result.effective_route == expected["effective_route"]
    assert result.companies == expected["companies"]
    assert result.vector_companies == expected["vector_companies"]
    assert result.years == expected["years"]
    for substring in expected["coverage_notes_contains"]:
        assert any(substring in note for note in result.notes), (
            f"expected a note containing {substring!r}, got {result.notes!r}"
        )
    if expected["clarification_expected"]:
        assert result.clarification is not None
        contains = expected.get("clarification_contains")
        if contains:
            assert contains in result.clarification
    else:
        assert result.clarification is None
    if "refusal_reason" in expected:
        assert result.refusal_reason == expected["refusal_reason"]


def test_resolve_company_unknown_returns_none(coverage):
    assert resolve_company("Netflix", coverage) is None


def test_resolve_company_alias_backstop(coverage):
    assert resolve_company("facebook", coverage) == "Meta"
    assert resolve_company("Alphabet", coverage) == "Google"
