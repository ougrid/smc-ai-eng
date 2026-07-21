"""sql_retrieve: ask the LLM to write one SQL query for the resolved
companies/years/metrics, validate + execute it via SqlTool (agent_ro role
only), then compute derived numbers (growth %) in Python -- the LLM only
structures the query, it never does the arithmetic. One retry on a
rejected/failed query, then flag the evidence as unavailable rather than
route around the gap.
"""

from typing import Any, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from app.agent.growth import compute_growth
from app.agent.sql_tool import SqlTool
from app.agent.state import AgentState

SQL_SYSTEM_PROMPT = """\
You write a single read-only PostgreSQL SELECT query against a table \
`financial_data(company, ticker, sector, year, revenue, gross_profit, \
operating_income, net_income)` (all figures in USD). Only ever reference \
`financial_data` -- no other tables, no CTEs referencing other tables, no \
joins to other tables. Respond with ONLY the SQL query: no explanation, \
no markdown code fences.\
"""


class SqlLLM(Protocol):
    def invoke(self, messages: list, config: RunnableConfig | None = None) -> Any: ...


def build_sql_llm(llm: BaseChatModel) -> SqlLLM:
    return llm.with_config(tags=["sql_retrieve"])


def _extract_sql(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _prompt_for(state: AgentState) -> str:
    companies = state.get("companies", [])
    years = state.get("years", [])
    route = state.get("route") or {}
    metrics = route.get("metrics", [])
    return (
        f"Question: {state['question']}\n"
        f"Companies (canonical names, use these exactly): {', '.join(companies) or 'unspecified'}\n"
        f"Years: {', '.join(map(str, years)) or 'unspecified'}\n"
        f"Metrics of interest: {', '.join(metrics) or 'unspecified'}\n"
    )


def build_sql_retrieve_node(sql_llm: SqlLLM, sql_tool: SqlTool):
    def _node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        messages: list[tuple[str, str]] = [
            ("system", SQL_SYSTEM_PROMPT),
            ("human", _prompt_for(state)),
        ]

        raw_sql = _extract_sql(sql_llm.invoke(messages, config=config).content)
        result = sql_tool.run(raw_sql)

        if result.error:
            messages.append(
                (
                    "human",
                    f"That query was rejected: {result.error}. Write one corrected "
                    "SELECT query, only from financial_data.",
                )
            )
            raw_sql = _extract_sql(sql_llm.invoke(messages, config=config).content)
            result = sql_tool.run(raw_sql)

        if result.error:
            notes = [*state.get("coverage_notes", []), "SQL evidence unavailable."]
            return {"sql": result.sql, "sql_rows": [], "computed": {}, "coverage_notes": notes}

        return {"sql": result.sql, "sql_rows": result.rows, "computed": compute_growth(result.rows)}

    return _node
