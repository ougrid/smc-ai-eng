"""capability: deterministic, no LLM call, no retrieval -- answers questions
about the assistant's OWN scope ("what can you help me with?", "which
companies' data do you have?", "what data do you have for 2024?").

Before this node existed, the route node's only options for a question with
no company/metric were "vague" (ask a clarifying question) or "off_topic"
(scope refusal) -- a plain onboarding question like "what data do you have?"
routinely got misclassified as off_topic and hit the apologetic refusal
template, which is bad UX for a completely legitimate, answerable question.
`capability` is its own intent (see nodes/route.py) precisely so this never
has to guess: it's answered directly, from the SAME live `CoverageMap` the
coverage gate itself uses, so the description can never drift from what
data is actually loaded (no hardcoded company counts/lists).

Bilingual, with light anti-repetition (same pattern as refuse.py): a second
capability question in one conversation gets an acknowledging variant
instead of a verbatim repeat.
"""

from app.agent.coverage import CoverageMap
from app.agent.state import AgentState


def _join_and(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _scope_summary(coverage: CoverageMap) -> dict:
    companies = sorted(coverage.sql_years.keys())
    years = sorted({y for ys in coverage.sql_years.values() for y in ys})
    vector_companies = sorted(coverage.vector_companies)
    # Always surface the 10-K-covered companies in the sample (most useful
    # to a user deciding what to ask), then pad with a few more household
    # names up to 8 total.
    sample = list(vector_companies)
    for c in companies:
        if len(sample) >= 8:
            break
        if c not in sample:
            sample.append(c)
    return {
        "count": len(companies),
        "sample": _join_and(sample),
        "remaining": max(0, len(companies) - len(sample)),
        "year_min": min(years) if years else "?",
        "year_max": max(years) if years else "?",
        "vector_companies": _join_and(vector_companies),
    }


_EN = [
    (
        "I can help with two kinds of financial questions: income-statement figures "
        "(revenue, gross profit, operating income, net income) for {year_min}-{year_max} "
        "across {count} U.S. public companies -- including {sample}, and {remaining} more "
        "-- and qualitative questions about FY2025 10-K filings, but only for {vector_companies}. "
        'Try something like "What was Apple\'s net income in 2024?" or "How does Google\'s '
        'revenue strategy compare to Meta\'s?"'
    ),
    (
        "As I mentioned, my coverage is the same: financial figures for {year_min}-{year_max} "
        "across {count} companies including {sample}, plus FY2025 10-K content for "
        "{vector_companies}. Ask about a specific company and metric -- say, Apple's net "
        "income trend, or how Meta's revenue strategy compares to Google's -- and I'll pull it up."
    ),
]

_TH = [
    (
        "ฉันช่วยตอบคำถามการเงินได้สองแบบค่ะ แบบแรกคือตัวเลขงบการเงิน (รายได้ กำไรขั้นต้น "
        "กำไรจากการดำเนินงาน กำไรสุทธิ) ปี {year_min}-{year_max} ของบริษัทมหาชนในสหรัฐฯ "
        "{count} แห่ง เช่น {sample} และอีก {remaining} แห่ง แบบที่สองคือคำถามเชิงคุณภาพจาก "
        "รายงาน 10-K ปี 2025 แต่มีเฉพาะ {vector_companies} เท่านั้น ลองถามเช่น "
        '"กำไรสุทธิของ Apple ปี 2024 เท่าไหร่" หรือ "กลยุทธ์รายได้ของ Google เทียบกับ Meta เป็นอย่างไร"'
    ),
    (
        "อย่างที่เรียนไปนะคะ ขอบเขตของฉันเหมือนเดิมค่ะ ตัวเลขงบการเงินปี {year_min}-{year_max} "
        "ของบริษัท {count} แห่ง เช่น {sample} รวมถึงเนื้อหาจากรายงาน 10-K ปี 2025 ของ "
        "{vector_companies} ลองถามเจาะจงบริษัทกับตัวเลขที่อยากรู้ได้เลยค่ะ"
    ),
]

_MARKERS_EN = ("two kinds of financial questions", "my coverage is the same")
_MARKERS_TH = ("ตอบคำถามการเงินได้สองแบบ", "ขอบเขตของฉันเหมือนเดิม")


def _language_of(state: AgentState) -> str:
    route = state.get("route") or {}
    language = route.get("language", "en")
    return "th" if str(language).lower().startswith("th") else "en"


def _prior_capability_count(state: AgentState) -> int:
    history = state.get("history") or []
    markers = _MARKERS_EN + _MARKERS_TH
    count = 0
    for role, text in history:
        if role != "assistant" or not isinstance(text, str):
            continue
        lowered = text.lower()
        if any(m in lowered for m in _MARKERS_EN) or any(m in text for m in _MARKERS_TH):
            count += 1
    return count


def build_capability_node(coverage: CoverageMap):
    summary = _scope_summary(coverage)

    def _node(state: AgentState) -> dict:
        language = _language_of(state)
        bank = _TH if language == "th" else _EN
        index = min(_prior_capability_count(state), len(bank) - 1)
        answer = bank[index].format(**summary)
        return {"final_answer": answer}

    return _node
