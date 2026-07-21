"""RAGAS-based quality regression suite -- a quantitative companion to
eval_baseline.py's binary pass/fail assertions. Scores faithfulness,
answer relevancy, and (where a reference exists) context recall/precision
for the three baseline questions (Take-Home Task.pdf), against a live
backend. Turns retrieval/grounding-quality regressions into numbers
instead of only catching outright breakage.

Requires the optional `eval` dependency group (not installed by default --
see README's "RAGAS quality-scoring eval (optional)" section for why): run
with `uv run --project backend --extra eval python scripts/eval_ragas.py`
(or `make eval-ragas`).

Ground-truth references for Q1 (Apple net income) and Q3 (revenue growth
ranking) are built from the same live-recomputed values eval_baseline.py
uses (never hardcoded). Q2 (qualitative revenue-structure/strategy
comparison) has no single numeric ground truth, so it's scored only on
the two reference-free metrics (faithfulness, answer relevancy) --
context recall/precision need a reference answer to compare against and
would be meaningless without one.

Not wired into CI (none exists yet in this repo) -- this is a standalone,
manually-run report. Costs a handful of gpt-4o-mini + embedding calls,
well under a cent.

Known limitation, observed and root-caused rather than guessed at:
`context_precision` reproducibly scores 0.000 for Q1's SQL-sourced
contexts. `ContextPrecisionWithReference` asks an LLM judge, per context,
a binary "was this context useful in arriving at THE reference" verdict,
then computes average precision from those verdicts (0 if every verdict
is 0). Q1's reference is one sentence spanning all four years; each
context is a single year's net_income row. The judge quite reasonably
marks each individual single-year context "not sufficient on its own"
for a reference describing all four years, driving every verdict to 0.
This is a real mismatch between how this metric is designed (single-fact
references against a mixed-relevance candidate pool) and this app's
actual SQL evidence shape (one row per fact, reference as a combined
summary) -- not a retrieval defect, and not something the pass/fail gate
below cares about (only `faithfulness` gates the exit code).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

import eval_baseline  # noqa: E402
from app.config import get_settings  # noqa: E402

# Faithfulness is this project's core no-hallucination signal -- a low score
# means the answer contains claims the retrieved evidence doesn't support.
# Lenient on purpose: LLM-judge scores carry real variance run to run: this
# should only ever fire on a genuine regression, not normal noise.
FAITHFULNESS_FLOOR = 0.5


def _contexts_from_metadata(metadata: dict[str, Any]) -> list[str]:
    """Chunk/SQL-row citations, plus Python-computed figures (growth %) --
    the latter are never literally present in any SQL row or chunk text
    (they're computed in agent/growth.py, not retrieved), so a synthesis
    answer stating "Meta grew 22.2%" has nothing in `citations` alone for
    a faithfulness judge to verify that specific claim against, even
    though the app's own deterministic verify node treats `computed` as
    part of the grounded evidence set (see agent/nodes/verify.py's
    `_grounded_pools`). Omitting it here would flag genuinely-grounded
    growth claims as unfaithful -- a false positive in the eval, not a
    real gap in the app."""
    contexts = []
    # SQL-row citations for a single-company query don't carry a "company"
    # key at all (the query was already scoped to it) -- without it, a
    # context string like "year=2022, net_income=99803000000" never
    # mentions Apple anywhere, so a context-precision judge comparing it
    # against a reference sentence about Apple's net income correctly
    # can't verify it's relevant. Reproduced as an exact, stable 0.000
    # score across repeated runs -- a missing-identifier bug in this
    # script's serialization, not a real retrieval-quality signal.
    gate_companies = (metadata.get("debug", {}).get("gate_result") or {}).get("companies") or []
    for c in metadata.get("citations", []):
        if "text" in c:
            contexts.append(c["text"])
        else:
            row = dict(c)
            if "company" not in row and gate_companies:
                row = {"company": "/".join(gate_companies), **row}
            contexts.append(", ".join(f"{k}={v}" for k, v in row.items() if k not in ("id", "score")))

    computed = metadata.get("debug", {}).get("computed") or {}
    for company, metrics in computed.items():
        for metric, value in metrics.items():
            contexts.append(f"{company} {metric.replace('_', ' ')}: {value}")

    return contexts


def _apple_net_income_reference(truth: dict[str, Any]) -> str:
    parts = [
        f"${value / 1e6:,.0f}M in {year}"
        for year, value in sorted(truth["apple_net_income"].items())
    ]
    return "Apple's net income was " + ", ".join(parts) + "."


def _growth_reference(truth: dict[str, Any]) -> str:
    meta_growth = truth["growth"]["Meta"]["revenue_growth_2024_2025"]
    return (
        f"Meta (Facebook) had the highest revenue growth from 2024 to 2025 among "
        f"Microsoft, Apple, Google, and Meta, at approximately {meta_growth:.1f}%."
    )


CASES = [
    # (name, question, reference-builder-or-None)
    (
        "Q1 (Apple net income, SQL)",
        "กำไรสุทธิของ Apple ปี 2022-2025 เป็นอย่างไร",
        _apple_net_income_reference,
    ),
    (
        "Q2 (Google vs Meta strategy, vector)",
        "เปรียบเทียบโครงสร้างรายได้และกลยุทธ์ทางธุรกิจของ Google และ Facebook ในปี 2025",
        None,
    ),
    (
        "Q3 (growth ranking, hybrid)",
        "จากรายได้ของ Microsoft, Apple, Google, Facebook ในปี 2024-2025 "
        "บริษัทใดมีอัตราการเติบโตสูงสุด และอะไรเป็นปัจจัยหลัก",
        _growth_reference,
    ),
]


def main() -> int:
    settings = get_settings()

    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.openai_chat_model, provider="openai", client=client)
    embeddings = OpenAIEmbeddings(client=client, model=settings.openai_embed_model)

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_recall = ContextRecall(llm=llm)
    context_precision = ContextPrecisionWithReference(llm=llm)

    truth = eval_baseline.ground_truth()

    with httpx.Client(base_url=eval_baseline.API_URL, timeout=eval_baseline.TIMEOUT) as http:
        for attempt in range(30):
            try:
                http.get("/api/health").raise_for_status()
                break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            print(f"FATAL: {eval_baseline.API_URL} not reachable -- is `make api` running?")
            return 1

        token = eval_baseline.register_throwaway_user(http)

        rows: list[dict[str, Any]] = []
        for name, question, reference_fn in CASES:
            result = eval_baseline.post_chat(http, token, question)
            answer = result["answer"]
            contexts = _contexts_from_metadata(result["metadata"])
            reference = reference_fn(truth) if reference_fn else None

            row: dict[str, Any] = {"case": name}
            row["faithfulness"] = faithfulness.score(
                user_input=question, response=answer, retrieved_contexts=contexts
            ).value
            row["answer_relevancy"] = answer_relevancy.score(
                user_input=question, response=answer
            ).value
            if reference is not None:
                row["context_recall"] = context_recall.score(
                    user_input=question, retrieved_contexts=contexts, reference=reference
                ).value
                row["context_precision"] = context_precision.score(
                    user_input=question, reference=reference, retrieved_contexts=contexts
                ).value
            rows.append(row)

    print()
    print("=" * 78)
    ok = True
    for row in rows:
        print(f"{row['case']}")
        for metric in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
            if metric in row:
                print(f"    {metric:<20} {row[metric]:.3f}")
        if row["faithfulness"] < FAITHFULNESS_FLOOR:
            print(f"    ^^ FAIL: faithfulness below floor ({FAITHFULNESS_FLOOR})")
            ok = False
        print()
    print("=" * 78)
    print("PASS" if ok else "FAIL: at least one case fell below the faithfulness floor")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
