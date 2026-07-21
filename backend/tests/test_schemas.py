"""Strict-json_schema compatibility for the LLM-facing structured models,
plus their field_validators. Compatibility here means what OpenAI's
strict mode requires: every property `required`, no bare `dict` fields,
`additionalProperties: false` at every level (including nested models).
"""

import pytest
from pydantic import ValidationError

from app.agent.schemas import CompanyMention, RouteDecision, SynthesisEnvelope


def _assert_strict_compatible(schema: dict):
    assert schema.get("additionalProperties") is False
    properties = schema.get("properties", {})
    assert set(schema.get("required", [])) == set(properties)
    for value in properties.values():
        assert value.get("type") != "object" or "$ref" in value or "properties" in value
    for definition in schema.get("$defs", {}).values():
        _assert_strict_compatible(definition)


def test_route_decision_schema_is_strict_compatible():
    _assert_strict_compatible(RouteDecision.model_json_schema())


def test_synthesis_envelope_schema_is_strict_compatible():
    _assert_strict_compatible(SynthesisEnvelope.model_json_schema())


def test_company_mention_schema_is_strict_compatible():
    _assert_strict_compatible(CompanyMention.model_json_schema())


def _valid_kwargs(**overrides):
    defaults = dict(
        reasoning="r",
        intent="financial",
        companies=[],
        years=[2024],
        metrics=[],
        route="sql",
        clarification=None,
        language="en",
    )
    defaults.update(overrides)
    return defaults


def test_empty_language_rejected():
    with pytest.raises(ValidationError):
        RouteDecision(**_valid_kwargs(language="   "))


def test_year_strings_are_coerced_to_int():
    decision = RouteDecision(**_valid_kwargs(years=["2024", 2025]))
    assert decision.years == [2024, 2025]


def test_unknown_top_level_field_rejected():
    with pytest.raises(ValidationError):
        RouteDecision(**_valid_kwargs(unexpected_field="nope"))
