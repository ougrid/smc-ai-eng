"""Deterministic coverage gate.

This is the last line of defense against hallucination: even a perfectly
well-behaved router LLM can't route around data that doesn't exist, because
this gate runs after it, is pure Python, and cannot be talked out of a
refusal. Canonical company keys are the SQL table's own names (Google,
Meta, Apple, Amazon, Microsoft, ...) -- not "Alphabet"/"Facebook", which is
how the 10-Ks and the questions refer to the same companies.

`apply_gate` takes anything shaped like `RouteDecisionLike` (see the
Protocols below) rather than importing `agent.schemas.RouteDecision`
directly, so this module has no dependency on the LLM-facing schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import Engine, text

# Static vector-availability set: only these four companies have an indexed
# 10-K (see docs/implementation-plan.md verified data facts).
VECTOR_COMPANIES = frozenset({"Apple", "Amazon", "Google", "Meta"})

# Deterministic backstop behind the LLM's own world-knowledge normalization
# (see the RouteDecision prompt contract in agent/schemas.py) -- pins the
# data-specific quirks (Google-not-Alphabet, Meta-not-Facebook) and keeps
# fixture-driven tests deterministic regardless of model drift.
ALIASES: dict[str, str] = {
    "alphabet": "Google",
    "facebook": "Meta",
    "fb": "Meta",
    "instagram": "Meta",
    "ig": "Meta",
    "กูเกิล": "Google",
    "กูเกิ้ล": "Google",
    "เฟซบุ๊ก": "Meta",
    "เฟสบุ๊ค": "Meta",
    "เฟสบุ๊ก": "Meta",
    "แอปเปิล": "Apple",
    "แอปเปิ้ล": "Apple",
    "ไมโครซอฟท์": "Microsoft",
    "ไมโครซอฟต์": "Microsoft",
    "อเมซอน": "Amazon",
    "แอมะซอน": "Amazon",
}


@dataclass(frozen=True)
class CoverageMap:
    sql_years: dict[str, list[int]]
    vector_companies: frozenset[str] = VECTOR_COMPANIES

    def known_company(self, name: str) -> bool:
        return name in self.sql_years

    def has_vector(self, name: str) -> bool:
        return name in self.vector_companies


def build_coverage(engine: Engine) -> CoverageMap:
    """Build the startup coverage map from the live `financial_data` table.

    Grouped in Python rather than `array_agg` so this runs unchanged against
    both the real Postgres engine and the in-memory SQLite engine tests use
    (see backend/tests/test_auth.py's stub-injection pattern).
    """
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT company, year FROM financial_data")).all()
    sql_years: dict[str, list[int]] = {}
    for company, year in rows:
        sql_years.setdefault(company, []).append(year)
    for years in sql_years.values():
        years.sort()
    return CoverageMap(sql_years=sql_years)


def resolve_company(name: str, coverage: CoverageMap) -> str | None:
    """Normalize a mention to a canonical SQL company name via the alias
    backstop. Returns None if unresolvable against our actual data --
    callers must treat that as "unknown", never guess further."""
    if coverage.known_company(name):
        return name
    canonical = ALIASES.get(name.strip().lower())
    if canonical and coverage.known_company(canonical):
        return canonical
    return None


class CompanyMentionLike(Protocol):
    mentioned: str
    canonical: str | None
    confident: bool


class RouteDecisionLike(Protocol):
    intent: str  # "financial" | "off_topic" | "vague" | "capability"
    companies: list[CompanyMentionLike]
    years: list[int]
    route: str  # "sql" | "vector" | "both" | "refuse" | "clarify" | "capability"
    clarification: str | None


@dataclass
class GateResult:
    effective_route: str  # "sql" | "vector" | "both" | "refuse" | "clarify" | "capability"
    companies: list[str] = field(default_factory=list)  # canonical, resolved
    vector_companies: list[str] = field(default_factory=list)  # subset with a 10-K
    years: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    clarification: str | None = None
    refusal_reason: str | None = None


def apply_gate(decision: RouteDecisionLike, coverage: CoverageMap) -> GateResult:
    if decision.intent == "capability":
        return GateResult(effective_route="capability")
    if decision.intent == "off_topic":
        return GateResult(effective_route="refuse", refusal_reason="out_of_scope")
    if decision.intent == "vague":
        return GateResult(effective_route="clarify", clarification=decision.clarification)

    resolved: list[str] = []
    unresolved_mentions: list[str] = []
    unknown_companies: list[str] = []
    for mention in decision.companies:
        if not mention.confident or mention.canonical is None:
            unresolved_mentions.append(mention.mentioned)
            continue
        canonical = resolve_company(mention.canonical, coverage)
        if canonical is None:
            unknown_companies.append(mention.canonical)
            continue
        if canonical not in resolved:
            resolved.append(canonical)

    if unresolved_mentions:
        # The LLM itself couldn't confidently resolve a mention -- ask,
        # never guess. Takes priority over unknown-company handling below.
        question = decision.clarification or (
            "Which company did you mean: " + ", ".join(unresolved_mentions) + "?"
        )
        return GateResult(effective_route="clarify", companies=resolved, clarification=question)

    notes: list[str] = []
    if unknown_companies:
        notes.append(f"No data available for: {', '.join(unknown_companies)}.")
    if not resolved:
        return GateResult(effective_route="refuse", refusal_reason="unknown_company", notes=notes)

    years = _trim_years(decision.years, resolved, coverage, notes)
    if decision.years and not years:
        return GateResult(
            effective_route="refuse",
            companies=resolved,
            refusal_reason="years_out_of_range",
            notes=notes,
        )

    route = decision.route
    vector_companies = [c for c in resolved if coverage.has_vector(c)]
    if route in ("vector", "both"):
        no_10k = [c for c in resolved if c not in vector_companies]
        for company in no_10k:
            notes.append(
                f'{company} has no 10-K filing indexed -- qualitative "why" '
                "cannot be grounded for it."
            )
        if route == "vector" and not vector_companies:
            return GateResult(
                effective_route="refuse",
                companies=resolved,
                years=years,
                refusal_reason="no_10k_for_company",
                notes=notes,
            )

    return GateResult(
        effective_route=route,
        companies=resolved,
        vector_companies=vector_companies,
        years=years,
        notes=notes,
    )


def _trim_years(
    years: list[int], companies: list[str], coverage: CoverageMap, notes: list[str]
) -> list[int]:
    if not years:
        return []
    valid: set[int] = set()
    for company in companies:
        available = set(coverage.sql_years.get(company, []))
        missing = sorted(y for y in years if y not in available)
        if missing:
            notes.append(f"{company} has no data for {', '.join(map(str, missing))}.")
        valid.update(y for y in years if y in available)
    return sorted(valid)
