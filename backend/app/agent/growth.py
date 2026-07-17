"""Growth-rate math, computed in Python from SQL rows -- never by the LLM
(see docs/implementation-plan.md item 3). Kept as a standalone pure
function so it's trivially unit-testable and reusable from both
sql_retrieve and eval_baseline.py.
"""

from typing import Any

METRIC_COLUMNS = ("revenue", "gross_profit", "operating_income", "net_income")


def compute_growth(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Group rows by company/year, then compute year-over-year percent
    growth for each metric column across every pair of consecutive years
    present in the result set. Rows missing `company`/`year`, or a metric
    that's null/zero in the earlier year, are skipped rather than raising --
    partial SQL result sets are the normal case (not every company has
    every field, e.g. Amazon's `gross_profit` is all NULL).
    """
    by_company: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        company, year = row.get("company"), row.get("year")
        if company is None or year is None:
            continue
        by_company.setdefault(company, {})[year] = row

    growth: dict[str, dict[str, float]] = {}
    for company, by_year in by_company.items():
        years = sorted(by_year)
        for prev_year, year in zip(years, years[1:]):
            for metric in METRIC_COLUMNS:
                before = by_year[prev_year].get(metric)
                after = by_year[year].get(metric)
                if not before or after is None:
                    continue
                pct = (after - before) / before * 100
                growth.setdefault(company, {})[f"{metric}_growth_{prev_year}_{year}"] = round(
                    pct, 2
                )
    return growth
