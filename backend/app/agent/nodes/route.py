"""Route node: one structured-output LLM call (intent classification +
world-knowledge entity resolution + route proposal), followed immediately
by the deterministic coverage gate (agent/coverage.py).

Fail-closed: a malformed LLM output gets exactly one re-ask, appending the
validation-error text to the conversation; a second failure routes to
refuse -- never to a guessed retrieval path. Strict json_schema mode only
guarantees *syntax*, so `message.refusal` and max-token truncation still
surface as `parsed=None`, same as a schema-validation failure; both are
handled identically here.
"""

from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.agent.coverage import CoverageMap, apply_gate
from app.agent.schemas import RouteDecision
from app.agent.state import AgentState

SYSTEM_PROMPT = """\
You are the router for a financial Q&A assistant grounded in two data \
sources: a SQL table of income-statement financials (~48 U.S. companies, \
2022-2025) and a vector store of FY2025 10-K filings (Alphabet, Amazon, \
Apple, Meta only). Classify the user's question and extract structured \
routing information.

intent:
- "financial": a question about a company's financials or 10-K content. A \
question naming one or more companies AND an analytical goal (compare, \
rank, growth, trend, "how is X doing", strengths/weaknesses) is "financial" \
even if it doesn't name an exact column -- infer the most relevant \
metric(s) yourself (e.g. a growth-rate/ranking question about named \
companies implies metrics=["revenue"] unless another figure is clearly \
meant). Only use "vague" when you genuinely cannot infer ANY company or \
ANY metric.
- "off_topic": unrelated to company financials (recipes, code, chit-chat).
- "vague": no company named AND no metric inferable (e.g. "How's the \
company doing?" with nothing else to go on) -- ask a targeted clarifying \
question in `clarification`.

metrics: one or more of revenue, gross_profit, operating_income, \
net_income -- infer these from context, including non-English phrasing \
(e.g. Thai "รายได้"/"อัตราการเติบโต" -> revenue, "กำไรสุทธิ" -> net_income). \
Example: "จากรายได้ของ Microsoft, Apple, Google, Facebook ในปี 2024-2025 \
บริษัทใดมีอัตราการเติบโตสูงสุด และอะไรเป็นปัจจัยหลัก" (revenue growth \
ranking + the "why") -> intent="financial" (NOT vague -- four companies \
and revenue growth are both present), metrics=["revenue"], route="both" \
(SQL for the growth numbers, vector for the "why").

For every company mentioned, normalize it to its canonical official name \
using your own knowledge of real-world brands (e.g. "Facebook"/"IG" -> \
Meta, the iPhone maker -> Apple, Alphabet -> Google, since this dataset \
lists Google's parent under the name "Google", not "Alphabet"). If you \
cannot confidently resolve a mention, set canonical=null and \
confident=false -- never guess silently, and set route="clarify" with a \
clarification question offering your best-guess candidates.

route: "sql" for quantitative questions, "vector" for qualitative/strategy \
questions, "both" for hybrid questions (numbers AND an explanation/"why"), \
"refuse" for off-topic intent, "clarify" for vague intent or any \
unconfident company mention. Always answer in the same language as the \
question (language = e.g. "en", "th").\
"""


class RouteLLM(Protocol):
    """The structured-output-bound runnable route_node expects -- i.e. the
    result of `llm.with_structured_output(RouteDecision, method="json_schema",
    strict=True, include_raw=True)`. Kept as a Protocol so tests can stub it
    without constructing a real LangChain chat model."""

    def invoke(self, messages: list, config: RunnableConfig | None = None) -> dict[str, Any]: ...


def build_route_llm(llm: BaseChatModel) -> RouteLLM:
    return llm.with_structured_output(
        RouteDecision, method="json_schema", strict=True, include_raw=True
    ).with_config(tags=["route"])


def _invoke_with_reask(
    route_llm: RouteLLM, messages: list[tuple[str, str]], config: RunnableConfig | None
) -> RouteDecision | None:
    result = route_llm.invoke(messages, config=config)
    if result["parsed"] is not None:
        return result["parsed"]

    error_text = str(result.get("parsing_error") or "response did not match the required schema")
    retry_messages = [
        *messages,
        (
            "human",
            f"Your previous response was invalid: {error_text}. "
            "Respond again, strictly matching the required schema.",
        ),
    ]
    result = route_llm.invoke(retry_messages, config=config)
    return result["parsed"]  # None on a second failure -> fail-closed


def build_route_node(coverage: CoverageMap, route_llm: RouteLLM):
    def _node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages = [("system", SYSTEM_PROMPT), ("human", state["question"])]
        decision = _invoke_with_reask(route_llm, messages, config)

        if decision is None:
            return {
                "route": None,
                "effective_route": "refuse",
                "companies": [],
                "vector_companies": [],
                "years": [],
                "metrics": [],
                "coverage_notes": [],
                "refusal_reason": "malformed_output",
                "clarification": None,
                "debug": {"route_decision": None, "gate_result": None},
            }

        gate = apply_gate(decision, coverage)
        return {
            "route": decision.model_dump(),
            "effective_route": gate.effective_route,
            "companies": gate.companies,
            "vector_companies": gate.vector_companies,
            "years": gate.years,
            "metrics": decision.metrics,
            "coverage_notes": gate.notes,
            "refusal_reason": gate.refusal_reason,
            "clarification": gate.clarification,
            "debug": {
                "route_decision": decision.model_dump(),
                "gate_result": vars(gate),
            },
        }

    return _node
