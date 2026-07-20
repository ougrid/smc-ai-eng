"""verify: deterministic, no-LLM numeric-consistency guard (see
docs/technical-execution-plan.md E6). Every number-like token in the draft
`final_answer` must be grounded in the retrieved evidence -- SQL rows
(at any of the raw/thousands/millions/billions scales a synthesis prompt
might present them at), Python-computed growth percentages, or a literal
match somewhere in cited chunk text. Formatting variants (commas, `$`, `%`)
and one-decimal-place rounding are allowed; a number that matches nothing
is flagged.

Fail-closed via the graph, not this node: `graph.py`'s conditional edge
retries `synthesize` once on failure, then routes to `refuse` -- this node
only reports what it found, it never decides how many attempts are left.
"""

import re
from typing import Any

from app.agent.state import AgentState

_NUMBER_RE = re.compile(r"\$?-?\d[\d,]*(?:\.\d+)?%?")
_TOLERANCE = 0.06  # one-decimal-place rounding slack
_SCALES = (1, 1_000, 1_000_000, 1_000_000_000)  # raw / thousands / millions / billions


def _parse_token(token: str) -> tuple[float, bool] | None:
    is_percent = token.endswith("%")
    cleaned = token.strip("$%").replace(",", "")
    if not cleaned or cleaned in ("-", "."):
        return None
    try:
        return float(cleaned), is_percent
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    # A bracketed citation marker like "[Meta_10K.pdf, p.12]" isn't a factual
    # claim -- strip the whole bracket (source filenames routinely embed
    # digits, e.g. the "10" in "10K") so it never gets flagged as ungrounded.
    text = re.sub(r"\[[^\]]*\]", "", text)
    out: list[tuple[str, float, bool]] = []
    for match in _NUMBER_RE.finditer(text):
        token = match.group()
        if not any(ch.isdigit() for ch in token):
            continue
        parsed = _parse_token(token)
        if parsed is not None:
            out.append((token, *parsed))
    return out


def _numeric_values(obj: Any) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [v for value in obj.values() for v in _numeric_values(value)]
    if isinstance(obj, list):
        return [v for value in obj for v in _numeric_values(value)]
    return []


def _grounded_pools(state: AgentState) -> tuple[set[float], set[float]]:
    """-> (plain_values, percent_values). SQL/row figures are stored at
    their raw dollar scale but a synthesis prompt is free to present them
    in $M or $B, so every scale variant is added to the plain pool;
    computed growth figures are already percentages."""
    plain: set[float] = set()
    for year in state.get("years") or []:
        plain.add(float(year))
    for value in _numeric_values(state.get("sql_rows")):
        for scale in _SCALES:
            plain.add(round(value / scale, 2))

    percents: set[float] = set()
    for value in _numeric_values(state.get("computed")):
        percents.add(round(value, 2))

    return plain, percents


def _within_tolerance(value: float, pool: set[float]) -> bool:
    if not pool:
        return False
    tolerance = max(_TOLERANCE, abs(value) * 0.005)
    return any(abs(value - g) <= tolerance for g in pool)


def _chunk_text(state: AgentState) -> str:
    return "\n".join(c.get("text", "") for c in (state.get("chunks") or []))


_CHUNK_SCALES = (1, 1_000)  # narrower than _SCALES (SQL's raw/M/M/B ladder):
# chunk-derived narrative numbers only realistically carry the single
# millions<->billions ambiguity 10-K prose introduces ("$196.6 billion" vs.
# a financial-statement table's "196,600"), never the full raw-to-billions
# range -- dividing a large fabricated number by 1e6/1e9 (as tried once,
# see git history) collapses it toward 0 or 1, and a degenerate "1"/"0"
# candidate then trivially substring-matches almost any prose, defeating
# the whole check.


def _literally_in_chunks(value: float, chunk_text: str) -> bool:
    """Still a strict "appears verbatim in the evidence" check, just
    scale-aware for the one ambiguity real 10-K prose introduces -- not a
    numeric-tolerance pool like `_grounded_pools`."""
    if not chunk_text:
        return False
    candidates: set[str] = set()
    for scale in _CHUNK_SCALES:
        for scaled in (value * scale, value / scale):
            if abs(scaled) < 1:
                continue  # e.g. 0.4 -> "0"/"1" would trivially match almost any text
            candidates.update(
                {f"{scaled:,.0f}", f"{scaled:,.1f}", f"{scaled:.0f}", f"{scaled:.1f}", f"{scaled:g}"}
            )
    return any(candidate in chunk_text for candidate in candidates)


def build_verify_node():
    def _node(state: AgentState) -> dict[str, Any]:
        answer = state.get("final_answer") or ""
        plain, percents = _grounded_pools(state)
        chunk_text = _chunk_text(state)

        ungrounded: list[str] = []
        for token, value, is_percent in _extract_numbers(answer):
            pool = percents if is_percent else plain
            other_pool = plain if is_percent else percents
            if _within_tolerance(value, pool) or _within_tolerance(value, other_pool):
                continue
            if _literally_in_chunks(value, chunk_text):
                continue
            ungrounded.append(token)

        attempt = state.get("verify_attempts", 0) + 1
        ok = not ungrounded
        result: dict[str, Any] = {"ok": ok, "ungrounded": ungrounded, "attempt": attempt}
        out: dict[str, Any] = {"verify": result, "verify_attempts": attempt}
        if not ok:
            out["refusal_reason"] = "unverified_numbers"
        return out

    return _node
