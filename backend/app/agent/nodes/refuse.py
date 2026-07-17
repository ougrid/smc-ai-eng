"""refuse: deterministic bilingual template, no LLM call. Three template
families, distinguished by `state["refusal_reason"]`: out-of-scope (the
intent gate caught an off-topic question), data-unavailable (the coverage
gate couldn't resolve a company, year range, or 10-K), and
unverified-numbers (the Day-4 `verify` node's numeric-consistency guard
failed twice -- the draft answer is discarded, never annotated, per
docs/technical-execution-plan.md E6).
"""

from app.agent.state import AgentState

_OUT_OF_SCOPE = {
    "en": (
        "I can only answer questions about U.S. public companies' financials "
        "and their FY2025 10-K filings (Alphabet, Amazon, Apple, Meta). For "
        'example: "What was Apple\'s net income from 2022-2025?" or "Compare '
        'Google and Meta\'s revenue strategy in 2025." Could you ask something '
        "in that scope?"
    ),
    "th": (
        "ฉันตอบได้เฉพาะคำถามเกี่ยวกับข้อมูลการเงินของบริษัทมหาชนในสหรัฐฯ และเนื้อหาจากรายงาน "
        "10-K ปี 2025 (Alphabet, Amazon, Apple, Meta) เท่านั้น เช่น \"กำไรสุทธิของ Apple "
        "ปี 2022-2025 เป็นอย่างไร\" หรือ \"เปรียบเทียบกลยุทธ์รายได้ของ Google กับ Meta ปี 2025\" "
        "ลองถามคำถามในขอบเขตนี้ได้ไหม"
    ),
}

_DATA_UNAVAILABLE = {
    "en": "I don't have the data to answer that.",
    "th": "ฉันไม่มีข้อมูลเพียงพอที่จะตอบคำถามนี้",
}

_UNVERIFIED_NUMBERS = {
    "en": (
        "I found a draft answer, but couldn't verify all the numbers in it "
        "against the retrieved data, so I can't give you a fully grounded "
        "response to this question."
    ),
    "th": (
        "ฉันพบคำตอบร่างไว้แล้ว แต่ไม่สามารถยืนยันตัวเลขทั้งหมดในคำตอบกับข้อมูลที่ดึงมาได้ "
        "จึงไม่สามารถให้คำตอบที่มีข้อมูลรองรับครบถ้วนสำหรับคำถามนี้ได้"
    ),
}


def _language_of(state: AgentState) -> str:
    route = state.get("route") or {}
    language = route.get("language", "en")
    return "th" if str(language).lower().startswith("th") else "en"


def refuse_node(state: AgentState) -> dict:
    language = _language_of(state)
    reason = state.get("refusal_reason")
    if reason == "out_of_scope":
        answer = _OUT_OF_SCOPE[language]
    elif reason == "unverified_numbers":
        answer = _UNVERIFIED_NUMBERS[language]
    else:
        notes = state.get("coverage_notes") or []
        answer = _DATA_UNAVAILABLE[language]
        if notes:
            answer = f"{answer} {' '.join(notes)}"
    return {"final_answer": answer}
