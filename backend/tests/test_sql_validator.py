"""sqlglot-based validator + SqlTool tests -- all offline against an
in-memory SQLite engine (validated SQL is plain ANSI SELECT/WHERE/GROUP
BY/LIMIT, so it runs unchanged against both SQLite and Postgres).
"""

import pytest
from sqlalchemy import create_engine, text

from app.agent.sql_tool import SqlTool, SqlValidationError, validate_sql


# --- validate_sql: accepted shapes ---


def test_accepts_plain_select():
    assert "financial_data" in validate_sql("SELECT * FROM financial_data")


def test_accepts_where_select():
    validate_sql("SELECT company, year FROM financial_data WHERE company = 'Apple'")


def test_accepts_aggregate_select():
    validate_sql("SELECT company, SUM(revenue) FROM financial_data GROUP BY company")


def test_injects_limit_when_missing():
    sql = validate_sql("SELECT * FROM financial_data", row_limit=100)
    assert "LIMIT 100" in sql.upper()


def test_caps_limit_above_row_limit():
    sql = validate_sql("SELECT * FROM financial_data LIMIT 100000", row_limit=100)
    assert "LIMIT 100" in sql.upper()
    assert "100000" not in sql


def test_keeps_limit_below_row_limit():
    sql = validate_sql("SELECT * FROM financial_data LIMIT 5", row_limit=100)
    assert "LIMIT 5" in sql.upper()


# --- validate_sql: rejected shapes ---


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE financial_data SET revenue = 0",
        "DELETE FROM financial_data",
        "DROP TABLE financial_data",
        "SELECT * FROM financial_data; DROP TABLE users;",
        "SELECT * FROM users",
        "SELECT * INTO new_table FROM financial_data",
        "SELECT * FROM financial_data WHERE company IN (SELECT email FROM users)",
        "WITH x AS (SELECT * FROM users) SELECT * FROM x",
        "SELECT * FROM financial_data f JOIN users u ON 1=1",
        "not even sql",
    ],
)
def test_rejects_disallowed_sql(sql):
    with pytest.raises(SqlValidationError):
        validate_sql(sql)


# --- SqlTool: execution against a seeded engine ---


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False})
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE financial_data (company TEXT, year INTEGER, revenue INTEGER)"))
        conn.execute(
            text("INSERT INTO financial_data VALUES ('Apple', 2024, 391035000000)")
        )
    return eng


def test_sql_tool_runs_validated_query(engine):
    result = SqlTool(engine).run("SELECT company, revenue FROM financial_data WHERE year = 2024")
    assert result.error is None
    assert result.rows == [{"company": "Apple", "revenue": 391035000000}]


def test_sql_tool_returns_error_for_rejected_query(engine):
    result = SqlTool(engine).run("SELECT * FROM users")
    assert result.error is not None
    assert result.rows == []
