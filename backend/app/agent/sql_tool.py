"""SQL validation and execution for the LLM-generated-SQL path.

Three guards defend this path (only the first two live in this module --
the third is the `agent_ro` Postgres role itself, see
scripts/initdb/02_roles.sql): (1) sqlglot validation below -- exactly one
SELECT statement, every referenced table (including CTE aliases and
subqueries) must be `financial_data`, no SELECT INTO, LIMIT injected/capped;
(2) the caller must connect via the agent_ro engine, never the app engine;
(3) `agent_ro` physically cannot read the `users` table at the Postgres
role level, so even a validator bug can't leak it.
"""

from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlalchemy import Engine, text
from sqlglot import exp

ALLOWED_TABLE = "financial_data"


class SqlValidationError(ValueError):
    pass


def validate_sql(sql: str, *, row_limit: int = 100) -> str:
    try:
        statements = [s for s in sqlglot.parse(sql, dialect="postgres") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SqlValidationError(f"could not parse SQL: {e}") from e

    if len(statements) != 1:
        raise SqlValidationError("exactly one SQL statement is allowed")

    stmt = statements[0]
    if not isinstance(stmt, exp.Select):
        raise SqlValidationError("only SELECT statements are allowed")

    if stmt.args.get("into") is not None:
        raise SqlValidationError("SELECT INTO is not allowed")

    tables = {t.name for t in stmt.find_all(exp.Table)}
    disallowed = tables - {ALLOWED_TABLE}
    if disallowed:
        raise SqlValidationError(
            f"query references disallowed table(s): {', '.join(sorted(disallowed))}"
        )

    limit_node = stmt.args.get("limit")
    current_limit: int | None = None
    if limit_node is not None:
        try:
            current_limit = int(limit_node.expression.this)
        except (AttributeError, ValueError):
            current_limit = None
    if limit_node is None or current_limit is None or current_limit > row_limit:
        stmt.set("limit", exp.Limit(expression=exp.Literal.number(row_limit)))

    return stmt.sql(dialect="postgres")


@dataclass
class SqlResult:
    sql: str
    rows: list[dict[str, Any]]
    error: str | None = None


class SqlTool:
    def __init__(self, engine: Engine, *, row_limit: int = 100):
        self._engine = engine
        self._row_limit = row_limit

    def run(self, sql: str) -> SqlResult:
        try:
            validated = validate_sql(sql, row_limit=self._row_limit)
        except SqlValidationError as e:
            return SqlResult(sql=sql, rows=[], error=str(e))

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(validated))
                rows = [dict(row._mapping) for row in result]
        except Exception as e:  # pragma: no cover -- defensive, DB-error path
            return SqlResult(sql=validated, rows=[], error=str(e))

        return SqlResult(sql=validated, rows=rows)
