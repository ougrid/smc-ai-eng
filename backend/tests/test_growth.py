from app.agent.growth import compute_growth


def test_computes_yoy_growth_for_consecutive_years():
    rows = [
        {"company": "Meta", "year": 2024, "revenue": 164501000000},
        {"company": "Meta", "year": 2025, "revenue": 201067000000},
    ]
    growth = compute_growth(rows)
    assert growth["Meta"]["revenue_growth_2024_2025"] == 22.23


def test_skips_rows_missing_company_or_year():
    rows = [{"company": None, "year": 2024, "revenue": 100}, {"year": 2025, "revenue": 200}]
    assert compute_growth(rows) == {}


def test_skips_metric_when_earlier_value_is_null_or_zero():
    rows = [
        {"company": "Amazon", "year": 2024, "gross_profit": None, "revenue": 100},
        {"company": "Amazon", "year": 2025, "gross_profit": 500, "revenue": 200},
    ]
    growth = compute_growth(rows)
    assert "gross_profit_growth_2024_2025" not in growth.get("Amazon", {})
    assert growth["Amazon"]["revenue_growth_2024_2025"] == 100.0


def test_handles_multiple_companies_independently():
    rows = [
        {"company": "Apple", "year": 2024, "revenue": 391035000000},
        {"company": "Apple", "year": 2025, "revenue": 416161000000},
        {"company": "Google", "year": 2024, "revenue": 350018000000},
        {"company": "Google", "year": 2025, "revenue": 402956000000},
    ]
    growth = compute_growth(rows)
    assert set(growth) == {"Apple", "Google"}
