"""synthesize: strict evidence-only contract. Quantitative claims must come
from SQL rows/computed growth figures only; qualitative claims must be
attributed with a [Source, p.N] citation matching a retrieved chunk;
coverage notes get reproduced verbatim where relevant; the answer is in
the question's language.

Empty evidence is a deterministic refusal handled *before* the LLM call --
cheaper and more reliable than trusting prompt discipline alone (the Day-4
`verify` node is the second, numeric-consistency line of defense on top of
this).

Bound the same way as the route node: `with_structured_output(...,
strict=True, include_raw=True)`, tagged "synthesize" so the SSE emitter can
filter its token stream to just this node's output; same
one-re-ask-then-refuse policy on `parsed=None`.

A capped history window (`agent/history.py`) is appended before the
question, same as the route node, but purely for phrasing continuity --
by the time this node runs, companies/years/evidence are already resolved
structurally by `route` + the coverage gate, so history here can't change
what's grounded, only how naturally the answer reads on a follow-up.
"""

from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.agent.history import history_messages
from app.agent.schemas import SynthesisEnvelope
from app.agent.state import AgentState

SYNTHESIS_SYSTEM_PROMPT = """\
You answer financial questions using ONLY the evidence provided below. \
Never invent numbers or facts. Quantitative claims must come only from the \
SQL rows / computed growth figures given. Qualitative claims must carry a \
[Source, p.N] citation matching a provided 10-K excerpt. Ignore print-to-PDF \
header/footer noise in excerpts (timestamps, file paths).

COVERAGE NOTES MUST APPEAR IN THE ANSWER TEXT ITSELF, not just be available \
as side metadata. Every coverage note given below (e.g. a company has no \
10-K indexed, or no substantive excerpts were retrieved for a company for \
this question) describes a real gap in what you can ground -- write one \
explicit sentence per such note as part of `answer`, naming the company and \
saying plainly that it cannot be grounded for the reason given. Do this even \
when it means the qualitative half of the answer is only a gap statement \
with no citation. Never let a coverage note exist only in the notes list \
while the answer text stays silent about it.

These are ACTUAL reported historical figures from SQL records and filed \
10-Ks, never projections or forecasts -- do not say "expected to have", \
"projected", or "forecast"; say what the figure IS or WAS.

ANSWER THE SPECIFIC QUESTION ASKED, don't just restate the evidence. If \
asked "which company/what is highest/what grew the most", name that \
company explicitly and state its number -- don't enumerate every company's \
data without concluding. If asked "why" and a company's why cannot be \
grounded (see coverage notes), say so explicitly for that company rather \
than omitting it.

TONE: write as a knowledgeable analyst talking to a colleague -- \
professional, clear, and informative, but conversational rather than stiff \
or robotic. Vary your sentence structure and lead with the substance; don't \
open every answer with the same boilerplate phrase ("Based on the data \
provided...", "According to the evidence..."). Get to the point naturally. \
None of this loosens the grounding rules above -- every number and every \
qualitative claim still comes only from the evidence, with citations.

INVESTMENT / ADVICE-FLAVORED QUESTIONS ("should I invest in X?", "is X a \
good buy?"): do NOT refuse these and do NOT give a buy/sell recommendation. \
Instead, give a grounded, balanced read built ONLY from the retrieved \
evidence -- the financial trends visible in the SQL rows/computed figures, \
and any strategy or risk factors from the 10-K excerpts (with citations). \
Present both the strengths and the weaknesses/risks the evidence actually \
shows; never manufacture a rosy or bleak picture the numbers don't support. \
NEVER invent forward-looking numbers, price targets, projections, or \
predictions -- state only what the figures ARE or WERE. Close with one brief, \
natural sentence noting this is data-based information, not personalized \
financial advice (do not pad it into a long disclaimer). If the question is \
investment-flavored but the qualitative side can't be grounded for a company \
(see coverage notes), say so plainly, same as any other question.

LANGUAGE (critical, check this last before responding): the human message \
below states a TARGET LANGUAGE explicitly -- write the ENTIRE `answer` in \
that exact language, even though the evidence (SQL rows, 10-K excerpts) is \
in English, and even if other examples in this system prompt are in a \
different language -- translate the substance, don't just copy English \
sentences, and don't switch language because the evidence or an example \
happens to be in one. Company names, tickers, and dollar figures may stay \
as-is.\
"""

_NO_EVIDENCE_ANSWER = "I don't have grounded data available to answer this question."
_MALFORMED_ANSWER = "I couldn't process this question -- please try rephrasing it."

_LANGUAGE_NAMES = {"en": "English", "th": "Thai"}


def _language_directive(state: AgentState) -> str:
    code = ((state.get("route") or {}).get("language") or "en").strip().lower()
    name = _LANGUAGE_NAMES.get(code, code)
    return f'TARGET LANGUAGE: {name} ("{code}"). Write `answer` entirely in {name}.'


class SynthesisLLM(Protocol):
    def invoke(self, messages: list, config: RunnableConfig | None = None) -> dict[str, Any]: ...


def build_synthesis_llm(llm: BaseChatModel) -> SynthesisLLM:
    return llm.with_structured_output(
        SynthesisEnvelope, method="json_schema", strict=True, include_raw=True
    ).with_config(tags=["synthesize"])


def _has_evidence(state: AgentState) -> bool:
    return bool(state.get("sql_rows") or state.get("chunks"))


def _format_evidence(state: AgentState) -> str:
    parts: list[str] = []
    if state.get("sql_rows"):
        parts.append(f"SQL rows: {state['sql_rows']}")
    if state.get("computed"):
        parts.append(f"Computed growth figures: {state['computed']}")
    if state.get("chunks"):
        lines = [f"[{c['source']}, p.{c['page']}] {c['text']}" for c in state["chunks"]]
        parts.append("10-K excerpts:\n" + "\n".join(lines))
    if state.get("coverage_notes"):
        parts.append(
            "Coverage notes (reproduce verbatim where relevant): "
            + "; ".join(state["coverage_notes"])
        )
    return "\n\n".join(parts)


def _invoke_with_reask(
    synth_llm: SynthesisLLM, messages: list[tuple[str, str]], config: RunnableConfig | None
) -> SynthesisEnvelope | None:
    result = synth_llm.invoke(messages, config=config)
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
    result = synth_llm.invoke(retry_messages, config=config)
    return result["parsed"]


def build_synthesize_node(synth_llm: SynthesisLLM, *, history_max_messages: int = 8):
    def _node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        if not _has_evidence(state):
            return {"envelope": None, "final_answer": _NO_EVIDENCE_ANSWER}

        human_message = (
            f"Question: {state['question']}\n\n" f"Evidence:\n{_format_evidence(state)}"
        )
        prior_verify = state.get("verify")
        if prior_verify and not prior_verify.get("ok"):
            hints: list[str] = []
            if prior_verify.get("ungrounded"):
                ungrounded = ", ".join(prior_verify["ungrounded"])
                hints.append(
                    f"Your previous answer included numbers that could not be verified "
                    f"against the evidence above: {ungrounded}. Regenerate the answer using "
                    "ONLY numbers that appear in the evidence -- drop or rephrase any claim "
                    "you cannot support instead of repeating an unverifiable figure."
                )
            if prior_verify.get("dangling_citations"):
                dangling = ", ".join(prior_verify["dangling_citations"])
                hints.append(
                    f"Your previous answer included a citation that did not match any "
                    f"retrieved evidence chunk: {dangling}. Regenerate using ONLY "
                    "[Source, p.N] citations that exactly match a chunk given in the "
                    "evidence above -- drop the citation, or the claim it supports, if "
                    "you cannot find a matching chunk."
                )
            for hint in hints:
                human_message += f"\n\n{hint}"
        # Last thing the model reads -- recency helps instruction-following,
        # and this is the exact bug the language directive exists to prevent
        # (see agent-output/day3-smoke-test-findings.md and the Day-4 fix:
        # an English question got a Thai answer when this was left implicit).
        human_message += f"\n\n{_language_directive(state)}"

        messages: list[tuple[str, str]] = [
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            *history_messages(state, history_max_messages),
            ("human", human_message),
        ]
        envelope = _invoke_with_reask(synth_llm, messages, config)

        if envelope is None:
            return {"envelope": None, "final_answer": _MALFORMED_ANSWER}

        return {"envelope": envelope.model_dump(), "final_answer": envelope.answer}

    return _node
