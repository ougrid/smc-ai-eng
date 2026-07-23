"""Structured-output contracts for the route and synthesize LLM calls.

Both models are bound via `with_structured_output(Model, method="json_schema",
strict=True, include_raw=True)` (see agent/nodes/route.py and synthesize.py).
OpenAI's strict json_schema mode requires every property to be `required` in
the schema -- nullability, not absence, is how "optional" is expressed --
and forbids additional properties, hence `extra="forbid"` and no bare
`dict` fields anywhere in this module. Field order is generation order on
gpt-4o-mini-class models, which is why `reasoning` is always first and
`answer` (the streamed field) precedes `citations`.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class CompanyMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mentioned: str  # verbatim from the question, e.g. "เฟซบุ๊ก", "the iPhone company"
    canonical: Optional[str]  # LLM's world-knowledge normalization; None = can't resolve
    confident: bool  # False -> gate sends to clarify with candidates


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str  # FIRST -- schema field order is generation order
    intent: Literal["financial", "off_topic", "vague", "capability"]
    companies: list[CompanyMention]  # world-knowledge entity resolution happens here
    years: list[int]
    metrics: list[str]
    route: Literal["sql", "vector", "both", "refuse", "clarify", "capability"]
    clarification: Optional[str]  # question to ask back; required key, nullable value
    language: str

    @field_validator("years", mode="before")
    @classmethod
    def _coerce_years(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [int(y) if isinstance(y, str) else y for y in value]

    @field_validator("language")
    @classmethod
    def _language_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("language must not be empty")
        return value


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["sql", "chunk"]
    source: Optional[str]
    page: Optional[int]
    quote: Optional[str]


class SynthesisEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str  # FIRST (discarded from the user-visible stream)
    answer: str  # SECOND -- the streamed field, see agent/sse.AnswerFieldExtractor
    citations: list[Citation]  # LAST -- arrives after answer closes, via data parts
