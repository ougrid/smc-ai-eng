"""refuse: deterministic bilingual template, no LLM call. Four template
families, distinguished by `state["refusal_reason"]`: out-of-scope (the
intent gate caught an off-topic question), data-unavailable (the coverage
gate couldn't resolve a company, year range, or 10-K), unverified-numbers
(the Day-4 `verify` node's numeric-consistency guard failed twice), and
unverified-citations (`verify`'s citation-resolution guard found a
`[Source, p.N]` marker that never resolved to a retrieved chunk) -- in
either verify failure case the draft answer is discarded, never annotated,
per docs/technical-execution-plan.md E6.

Anti-parroting: each family carries 2-3 phrasing variants per language. The
variant is chosen DETERMINISTICALLY from how many prior assistant refusals
already appear in `state["history"]` -- no LLM call, no invented facts, so
the no-hallucination contract is untouched. First refusal gets the plain
variant; a repeat acknowledges it naturally ("As I mentioned...") and steers
the user toward what IS answerable (SQL: ~48 U.S. companies 2022-2025;
10-Ks: Alphabet/Amazon/Apple/Meta FY2025). The variant index is clamped to
the last entry, so a third+ refusal reuses the most explicit steer rather
than overflowing.
"""

from app.agent.state import AgentState

_OUT_OF_SCOPE = {
    "en": [
        (
            "I can only answer questions about U.S. public companies' financials "
            "and their FY2025 10-K filings (Alphabet, Amazon, Apple, Meta). For "
            'example: "What was Apple\'s net income from 2022-2025?" or "Compare '
            'Google and Meta\'s revenue strategy in 2025." Could you ask something '
            "in that scope?"
        ),
        (
            "As I mentioned, that's outside what I can help with -- I'm limited to "
            "U.S. public companies' financials (2022-2025) and the FY2025 10-K "
            "filings for Alphabet, Amazon, Apple, and Meta. I'd be glad to dig into "
            "any of those, say Apple's net income trend or how Google and Meta "
            "compare on revenue."
        ),
        (
            "I still can't take that one on -- my scope is strictly the financial "
            "figures for ~48 U.S. public companies (2022-2025) and the FY2025 10-K "
            "filings for Alphabet, Amazon, Apple, and Meta. If any of those work "
            "for you, just point me at a company and a metric and I'll pull it up."
        ),
    ],
    "th": [
        (
            "ฉันตอบได้เฉพาะคำถามเกี่ยวกับข้อมูลการเงินของบริษัทมหาชนในสหรัฐฯ และเนื้อหาจากรายงาน "
            "10-K ปี 2025 (Alphabet, Amazon, Apple, Meta) เท่านั้น เช่น \"กำไรสุทธิของ Apple "
            "ปี 2022-2025 เป็นอย่างไร\" หรือ \"เปรียบเทียบกลยุทธ์รายได้ของ Google กับ Meta ปี 2025\" "
            "ลองถามคำถามในขอบเขตนี้ได้ไหม"
        ),
        (
            "อย่างที่เรียนไปนะคะ คำถามนี้อยู่นอกขอบเขตที่ฉันช่วยได้ ฉันดูแลเฉพาะข้อมูลการเงินของบริษัทมหาชน"
            "ในสหรัฐฯ (ปี 2022-2025) และเนื้อหาจากรายงาน 10-K ปี 2025 ของ Alphabet, Amazon, Apple "
            "และ Meta ถ้าสนใจเรื่องเหล่านี้ บอกชื่อบริษัทกับตัวเลขที่อยากรู้ได้เลย เช่น กำไรสุทธิของ Apple "
            "หรือเปรียบเทียบรายได้ของ Google กับ Meta"
        ),
        (
            "ฉันยังตอบคำถามนี้ให้ไม่ได้จริงๆ เพราะอยู่นอกขอบเขตข้อมูลที่มีค่ะ สิ่งที่ช่วยได้คือตัวเลขการเงิน"
            "ของบริษัทมหาชนในสหรัฐฯ ราว 48 แห่ง (ปี 2022-2025) และรายงาน 10-K ปี 2025 ของ "
            "Alphabet, Amazon, Apple และ Meta ถ้ามีบริษัทใดในกลุ่มนี้ที่อยากรู้ บอกได้เลยค่ะ"
        ),
    ],
}

# data-unavailable is composed as `lead + coverage notes`; only the lead
# sentence varies. The coverage notes (e.g. "No data available for: X.") are
# still appended verbatim after whichever lead is chosen.
_DATA_UNAVAILABLE = {
    "en": [
        "I don't have the data to answer that.",
        (
            "As I mentioned, I still don't have data for that -- but I can help with "
            "financials for ~48 U.S. public companies (2022-2025) and the FY2025 "
            "10-Ks for Alphabet, Amazon, Apple, and Meta."
        ),
        (
            "That's still not something I have data for. Where I can help is the "
            "financials of ~48 U.S. public companies across 2022-2025, plus the "
            "FY2025 10-K filings for Alphabet, Amazon, Apple, and Meta -- happy to "
            "pull any of those."
        ),
    ],
    "th": [
        "ฉันไม่มีข้อมูลเพียงพอที่จะตอบคำถามนี้",
        (
            "อย่างที่เรียนไปนะคะ ฉันยังไม่มีข้อมูลสำหรับคำถามนี้ แต่ฉันช่วยเรื่องข้อมูลการเงินของบริษัทมหาชน"
            "ในสหรัฐฯ ราว 48 แห่ง (ปี 2022-2025) และรายงาน 10-K ปี 2025 ของ Alphabet, Amazon, "
            "Apple และ Meta ได้"
        ),
        (
            "เรื่องนี้ยังเป็นข้อมูลที่ฉันไม่มีค่ะ สิ่งที่ช่วยได้คือตัวเลขการเงินของบริษัทมหาชนในสหรัฐฯ ราว 48 แห่ง "
            "(ปี 2022-2025) และรายงาน 10-K ปี 2025 ของ Alphabet, Amazon, Apple และ Meta"
        ),
    ],
}

_UNVERIFIED_NUMBERS = {
    "en": [
        (
            "I found a draft answer, but couldn't verify all the numbers in it "
            "against the retrieved data, so I can't give you a fully grounded "
            "response to this question."
        ),
        (
            "As I flagged a moment ago, I put together a draft but still couldn't "
            "reconcile every figure against the retrieved data, so I'd rather not "
            "give you a half-grounded answer. You might get further by narrowing to "
            "a specific company and year -- say, Apple's net income for 2024."
        ),
        (
            "Same issue again -- the draft I generated had numbers I couldn't tie "
            "back to the source data, and I won't pass along figures I can't stand "
            "behind. A tighter question (one company, one metric, one year within "
            "2022-2025) is the most reliable path."
        ),
    ],
    "th": [
        (
            "ฉันพบคำตอบร่างไว้แล้ว แต่ไม่สามารถยืนยันตัวเลขทั้งหมดในคำตอบกับข้อมูลที่ดึงมาได้ "
            "จึงไม่สามารถให้คำตอบที่มีข้อมูลรองรับครบถ้วนสำหรับคำถามนี้ได้"
        ),
        (
            "อย่างที่เพิ่งแจ้งไป ฉันร่างคำตอบได้ แต่ยังยืนยันตัวเลขทั้งหมดกับข้อมูลที่ดึงมาไม่ได้ "
            "จึงไม่อยากให้คำตอบที่ข้อมูลรองรับไม่ครบ ลองถามให้เจาะจงขึ้น เช่น กำไรสุทธิของ Apple ปี 2024 "
            "น่าจะได้ผลดีกว่าค่ะ"
        ),
        (
            "ยังติดปัญหาเดิมค่ะ ตัวเลขในคำตอบร่างยังเทียบกลับไปที่ข้อมูลต้นทางไม่ได้ และฉันจะไม่ส่งต่อตัวเลข"
            "ที่ยืนยันไม่ได้ ถ้าถามแบบเจาะจง (บริษัทเดียว ตัวเลขเดียว ปีเดียวในช่วง 2022-2025) จะแม่นยำที่สุด"
        ),
    ],
}

_UNVERIFIED_CITATIONS = {
    "en": [
        (
            "I found a draft answer, but one of its citations didn't match any of "
            "the retrieved evidence, so I can't give you a fully grounded response "
            "to this question."
        ),
        (
            "As I mentioned, the draft I generated cited a source I couldn't match "
            "to the retrieved evidence, so I can't stand behind it. If you narrow "
            "the question to one of the covered filings -- Alphabet, Amazon, Apple, "
            "or Meta's FY2025 10-K -- I can be more precise."
        ),
        (
            "Same problem as before -- a citation in my draft pointed to something "
            "not in the retrieved evidence, and I won't hand you a claim I can't "
            "source. Focusing on a single company's FY2025 10-K (Alphabet, Amazon, "
            "Apple, or Meta) tends to work best."
        ),
    ],
    "th": [
        (
            "ฉันพบคำตอบร่างไว้แล้ว แต่การอ้างอิงในคำตอบไม่ตรงกับข้อมูลที่ดึงมาได้ "
            "จึงไม่สามารถให้คำตอบที่มีข้อมูลรองรับครบถ้วนสำหรับคำถามนี้ได้"
        ),
        (
            "อย่างที่เรียนไปนะคะ คำตอบร่างที่ได้มีการอ้างอิงที่ไม่ตรงกับข้อมูลที่ดึงมา ฉันจึงไม่มั่นใจพอจะส่งให้ "
            "ลองถามเจาะจงไปที่รายงาน 10-K ปี 2025 ของ Alphabet, Amazon, Apple หรือ Meta "
            "จะช่วยให้แม่นยำขึ้นค่ะ"
        ),
        (
            "ยังเป็นปัญหาเดิมค่ะ การอ้างอิงในคำตอบร่างชี้ไปยังข้อมูลที่ไม่มีอยู่จริง และฉันจะไม่ส่งต่อข้อความ"
            "ที่อ้างอิงไม่ได้ การโฟกัสไปที่รายงาน 10-K ปี 2025 ของบริษัทใดบริษัทหนึ่ง (Alphabet, Amazon, "
            "Apple หรือ Meta) มักได้ผลดีที่สุด"
        ),
    ],
}

# Distinctive substrings, one per variant above, used to recognize a prior
# assistant refusal replayed in `state["history"]`. Chosen to be phrases that
# occur in refusals but not in normal grounded answers, so counting them does
# not misfire on ordinary turns. English is matched case-insensitively.
_REFUSAL_MARKERS_EN = (
    "questions about u.s. public companies' financials",
    "outside what i can help with",
    "i still can't take that one on",
    "don't have the data to answer that",
    "still don't have data for that",
    "still not something i have data for",
    "couldn't verify all the numbers",
    "couldn't reconcile every figure",
    "numbers i couldn't tie",
    "citations didn't match",
    "cited a source i couldn't match",
    "a citation in my draft pointed to",
)
_REFUSAL_MARKERS_TH = (
    "ตอบได้เฉพาะคำถามเกี่ยวกับข้อมูลการเงิน",
    "อยู่นอกขอบเขตที่ฉันช่วยได้",
    "ยังตอบคำถามนี้ให้ไม่ได้จริงๆ",
    "ไม่มีข้อมูลเพียงพอที่จะตอบคำถามนี้",
    "ยังไม่มีข้อมูลสำหรับคำถามนี้",
    "ยังเป็นข้อมูลที่ฉันไม่มี",
    "ไม่สามารถยืนยันตัวเลขทั้งหมด",
    "ยืนยันตัวเลขทั้งหมดกับข้อมูลที่ดึงมาไม่ได้",
    "เทียบกลับไปที่ข้อมูลต้นทางไม่ได้",
    "การอ้างอิงในคำตอบไม่ตรงกับข้อมูล",
    "การอ้างอิงที่ไม่ตรงกับข้อมูลที่ดึงมา",
    "การอ้างอิงในคำตอบร่างชี้ไปยังข้อมูลที่ไม่มีอยู่จริง",
)


def _language_of(state: AgentState) -> str:
    route = state.get("route") or {}
    language = route.get("language", "en")
    return "th" if str(language).lower().startswith("th") else "en"


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS_EN):
        return True
    return any(marker in text for marker in _REFUSAL_MARKERS_TH)


def _prior_refusal_count(state: AgentState) -> int:
    """How many prior assistant turns in this conversation were refusals.
    Drives variant selection so a user never sees the exact same refusal text
    twice in a row."""
    history = state.get("history") or []
    return sum(
        1
        for role, text in history
        if role == "assistant" and isinstance(text, str) and _looks_like_refusal(text)
    )


def _variant(bank: dict[str, list[str]], language: str, index: int) -> str:
    variants = bank[language]
    return variants[min(index, len(variants) - 1)]


def refuse_node(state: AgentState) -> dict:
    language = _language_of(state)
    reason = state.get("refusal_reason")
    index = _prior_refusal_count(state)

    if reason == "out_of_scope":
        answer = _variant(_OUT_OF_SCOPE, language, index)
    elif reason == "unverified_numbers":
        answer = _variant(_UNVERIFIED_NUMBERS, language, index)
    elif reason == "unverified_citations":
        answer = _variant(_UNVERIFIED_CITATIONS, language, index)
    else:
        notes = state.get("coverage_notes") or []
        answer = _variant(_DATA_UNAVAILABLE, language, index)
        if notes:
            answer = f"{answer} {' '.join(notes)}"
    return {"final_answer": answer}
