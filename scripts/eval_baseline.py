"""Live-stack integration eval for the three baseline questions
(Take-Home Task.pdf) plus a handful of refusal/clarify probes -- run
against a real running backend (`make api`), not stubs.

Registers a throwaway user, POSTs each Thai baseline question to
`/api/chat`, and consumes the SSE stream exactly as a real client would:
reassembling only the LAST text part per message (the stream-veto rule,
see app/agent/sse.py) and reading the `finish` event's `messageMetadata`
for route/citations/coverage_notes/debug. This exercises the full
streaming + verify pipeline end to end, not just the graph in isolation.

Ground-truth growth figures are recomputed from data/financial_data.sql at
run time (never hardcoded) via the same `compute_growth` the agent itself
uses, so the eval can't silently drift from the shipped data.

Usage: uv run --project backend python scripts/eval_baseline.py
       (or `make eval` / `python scripts\\eval_baseline.py` on Windows)
Exits 0 if every case passes, 1 otherwise.
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.agent.growth import compute_growth  # noqa: E402

API_URL = os.environ.get("EVAL_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(60.0, connect=5.0)


class CaseFailure(AssertionError):
    pass


def _parse_financial_data_sql(path: Path) -> list[dict[str, Any]]:
    """Parse the `COPY financial_data (...) FROM stdin;` block directly --
    no DB connection needed, so this works even before the app has ever
    queried Postgres. `\\N` (Postgres NULL) becomes Python None."""
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"COPY financial_data \(([^)]*)\) FROM stdin;\n(.*?)\n\\\.", text, re.DOTALL
    )
    if not match:
        raise RuntimeError(f"could not find COPY block in {path}")
    columns = [c.strip() for c in match.group(1).split(",")]
    rows = []
    for line in match.group(2).splitlines():
        if not line.strip():
            continue
        values = line.split("\t")
        row: dict[str, Any] = {}
        for col, raw in zip(columns, values):
            if raw == r"\N":
                row[col] = None
            elif col in ("year",) or col in (
                "revenue",
                "net_income",
                "operating_income",
                "gross_profit",
            ):
                row[col] = int(raw)
            else:
                row[col] = raw
        rows.append(row)
    return rows


def ground_truth() -> dict[str, Any]:
    rows = _parse_financial_data_sql(REPO_ROOT / "data" / "financial_data.sql")
    growth = compute_growth(rows)
    apple_net_income = {
        r["year"]: r["net_income"] for r in rows if r["company"] == "Apple"
    }
    return {"growth": growth, "apple_net_income": apple_net_income}


def register_throwaway_user(client: httpx.Client) -> str:
    email = f"eval-{uuid.uuid4().hex[:12]}@example.com"
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": "eval-baseline-pw", "display_name": "Eval Baseline"},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _sse_events(response: httpx.Response):
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            return
        import json

        yield json.loads(payload)


def post_chat(client: httpx.Client, token: str, question: str) -> dict[str, Any]:
    """POST one question, consume the SSE stream, and return
    {"answer": str, "metadata": dict} -- `answer` is only ever the LAST
    text part (mirrors the frontend's stream-veto rendering rule, so a
    fabricated draft that got corrected never counts as the answer here)."""
    body = {
        "id": str(uuid.uuid4()),
        "messages": [
            {"id": str(uuid.uuid4()), "role": "user", "parts": [{"type": "text", "text": question}]}
        ],
    }
    answer_parts: dict[str, list[str]] = {}
    current_id: str | None = None
    metadata: dict[str, Any] = {}

    with client.stream(
        "POST", "/api/chat", json=body, headers={"Authorization": f"Bearer {token}"}
    ) as response:
        response.raise_for_status()
        for event in _sse_events(response):
            kind = event.get("type")
            if kind == "text-start":
                current_id = event["id"]
                answer_parts.setdefault(current_id, [])
            elif kind == "text-delta":
                answer_parts.setdefault(event["id"], []).append(event["delta"])
            elif kind == "finish":
                metadata = event.get("messageMetadata", {})

    if not answer_parts:
        return {"answer": "", "metadata": metadata}
    last_id = list(answer_parts.keys())[-1]
    return {"answer": "".join(answer_parts[last_id]), "metadata": metadata}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CaseFailure(message)


def _assert_no_fabricated_numbers(metadata: dict[str, Any]) -> None:
    verify = metadata.get("verify")
    if verify is not None:
        _require(verify.get("ok") is True, f"verify failed: {verify.get('ungrounded')}")


# --- baseline cases -----------------------------------------------------


def case_q1_apple_net_income(client: httpx.Client, token: str, truth: dict[str, Any]) -> None:
    result = post_chat(client, token, "กำไรสุทธิของ Apple ปี 2022-2025 เป็นอย่างไร")
    metadata = result["metadata"]
    _require(metadata.get("route") == "sql", f"expected route=sql, got {metadata.get('route')}")
    _assert_no_fabricated_numbers(metadata)

    sql_rows = [c for c in metadata.get("citations", []) if "net_income" in c]
    got = {row["year"]: row["net_income"] for row in sql_rows if row.get("year") is not None}
    for year, expected in truth["apple_net_income"].items():
        _require(
            got.get(year) == expected,
            f"Apple net_income {year}: expected {expected}, got {got.get(year)} (rows={sql_rows})",
        )


def case_q2_google_vs_meta_strategy(client: httpx.Client, token: str) -> None:
    result = post_chat(
        client,
        token,
        "เปรียบเทียบโครงสร้างรายได้และกลยุทธ์ทางธุรกิจของ Google และ Facebook ในปี 2025",
    )
    metadata = result["metadata"]
    _require(
        metadata.get("route") in ("vector", "both"),
        f"expected route in (vector, both), got {metadata.get('route')}",
    )
    _assert_no_fabricated_numbers(metadata)

    chunks = [c for c in metadata.get("citations", []) if "company" in c and "text" in c]
    companies = {c["company"] for c in chunks}
    _require("Google" in companies, f"no Google chunk citations found (companies={companies})")
    _require("Meta" in companies, f"no Meta chunk citations found (companies={companies})")
    _require(
        all(c.get("page") is not None for c in chunks),
        "every chunk citation should carry a page number",
    )


def case_q3_growth_ranking(client: httpx.Client, token: str, truth: dict[str, Any]) -> None:
    result = post_chat(
        client,
        token,
        "จากรายได้ของ Microsoft, Apple, Google, Facebook ในปี 2024-2025 "
        "บริษัทใดมีอัตราการเติบโตสูงสุด และอะไรเป็นปัจจัยหลัก",
    )
    metadata = result["metadata"]
    _require(metadata.get("route") == "both", f"expected route=both, got {metadata.get('route')}")
    _assert_no_fabricated_numbers(metadata)

    meta_growth = truth["growth"].get("Meta", {}).get("revenue_growth_2024_2025")
    _require(meta_growth is not None, "ground truth is missing Meta's 2024->2025 revenue growth")
    _require(meta_growth > 20, f"sanity check failed: Meta growth {meta_growth}% is not ~22%")

    answer = result["answer"]
    _require("Meta" in answer or "Facebook" in answer, "answer doesn't name Meta/Facebook")
    coverage_notes = " ".join(metadata.get("coverage_notes", []))
    _require(
        "Microsoft" in coverage_notes and "10-K" in coverage_notes,
        f"expected a Microsoft-has-no-10-K coverage note, got: {coverage_notes!r}",
    )


# --- refusal / clarify probes --------------------------------------------


def case_blackrock_year_out_of_range(client: httpx.Client, token: str) -> None:
    result = post_chat(client, token, "What was BlackRock's net income in 2025?")
    metadata = result["metadata"]
    _require(
        metadata.get("route") == "refuse",
        f"expected route=refuse for BlackRock 2025, got {metadata.get('route')}",
    )


def case_netflix_no_10k(client: httpx.Client, token: str) -> None:
    result = post_chat(client, token, "What is Netflix's content strategy according to its 10-K?")
    metadata = result["metadata"]
    _require(
        metadata.get("route") == "refuse",
        f"expected route=refuse for Netflix qualitative question, got {metadata.get('route')}",
    )


def case_siemens_unknown_company(client: httpx.Client, token: str) -> None:
    result = post_chat(client, token, "What was Siemens' net income in 2025?")
    metadata = result["metadata"]
    _require(
        metadata.get("route") == "refuse",
        f"expected route=refuse for Siemens (unknown company), got {metadata.get('route')}",
    )


def case_off_topic_scope_refusal(client: httpx.Client, token: str) -> None:
    result = post_chat(client, token, "แนะนำร้านอาหารหน่อย")
    metadata = result["metadata"]
    _require(
        metadata.get("route") == "refuse",
        f"expected route=refuse for off-topic question, got {metadata.get('route')}",
    )


def case_vague_clarify(client: httpx.Client, token: str) -> None:
    result = post_chat(client, token, "บริษัทเป็นยังไงบ้าง")
    metadata = result["metadata"]
    _require(
        metadata.get("route") == "clarify",
        f"expected route=clarify for vague question, got {metadata.get('route')}",
    )


CASES = [
    ("Q1: Apple net income 2022-2025 (SQL)", case_q1_apple_net_income, True),
    ("Q2: Google vs Meta revenue structure/strategy 2025 (vector)", case_q2_google_vs_meta_strategy, False),
    ("Q3: highest revenue growth 2024-2025 + why (hybrid)", case_q3_growth_ranking, True),
    ("Refusal: BlackRock 2025 (year out of range)", case_blackrock_year_out_of_range, False),
    ("Refusal: Netflix 10-K strategy (no 10-K indexed)", case_netflix_no_10k, False),
    ("Refusal: Siemens (unknown company)", case_siemens_unknown_company, False),
    ("Refusal: off-topic question (scope gate)", case_off_topic_scope_refusal, False),
    ("Clarify: vague question", case_vague_clarify, False),
]


def main() -> int:
    truth = ground_truth()

    with httpx.Client(base_url=API_URL, timeout=TIMEOUT) as client:
        for attempt in range(30):
            try:
                client.get("/api/health").raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            print(f"FATAL: {API_URL} not reachable -- is `make api` running?")
            return 1

        token = register_throwaway_user(client)

        results: list[tuple[str, bool, str]] = []
        for name, case_fn, needs_truth in CASES:
            try:
                if needs_truth:
                    case_fn(client, token, truth)
                else:
                    case_fn(client, token)
                results.append((name, True, ""))
            except CaseFailure as e:
                results.append((name, False, str(e)))
            except httpx.HTTPError as e:
                results.append((name, False, f"HTTP error: {e}"))

    print()
    print("=" * 70)
    passed = 0
    for name, ok, message in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}")
        if not ok:
            print(f"       {message}")
        passed += ok
    print("=" * 70)
    print(f"{passed}/{len(results)} passed")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
