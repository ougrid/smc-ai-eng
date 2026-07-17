"""sql_retrieve node tests -- offline. The SQL LLM is stubbed directly
(text-in, text-out, matching a plain chat-model `.invoke().content`
interface); SqlTool runs for real against a seeded in-memory SQLite engine
so the validator/execution path is exercised end-to-end.
"""

from sqlalchemy import create_engine, text

from app.agent.nodes.sql_retrieve import build_sql_retrieve_node
from app.agent.sql_tool import SqlTool


class _Message:
    def __init__(self, content):
        self.content = content


class _StubSqlLLM:
    def __init__(self, contents):
        self._contents = list(contents)
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1
        return _Message(self._contents.pop(0))


def _seeded_sql_tool():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE financial_data (company TEXT, year INTEGER, revenue INTEGER)"))
        conn.execute(text("INSERT INTO financial_data VALUES ('Apple', 2024, 391035000000)"))
        conn.execute(text("INSERT INTO financial_data VALUES ('Apple', 2025, 416161000000)"))
    return SqlTool(engine)


def _state(**overrides):
    base = {"question": "Apple revenue growth", "companies": ["Apple"], "years": [2024, 2025]}
    base.update(overrides)
    return base


def test_extracts_sql_from_markdown_fence_and_runs_it():
    llm = _StubSqlLLM(
        ["```sql\nSELECT company, year, revenue FROM financial_data WHERE company = 'Apple'\n```"]
    )
    node = build_sql_retrieve_node(llm, _seeded_sql_tool())
    result = node(_state(), {})
    assert len(result["sql_rows"]) == 2
    assert result["computed"]["Apple"]["revenue_growth_2024_2025"] == 6.43
    assert llm.calls == 1


def test_retries_once_after_a_rejected_query():
    llm = _StubSqlLLM(
        [
            "SELECT * FROM users",
            "SELECT company, year, revenue FROM financial_data",
        ]
    )
    node = build_sql_retrieve_node(llm, _seeded_sql_tool())
    result = node(_state(), {})
    assert len(result["sql_rows"]) == 2
    assert llm.calls == 2


def test_flags_evidence_unavailable_after_second_failure():
    llm = _StubSqlLLM(["SELECT * FROM users", "SELECT * FROM users"])
    node = build_sql_retrieve_node(llm, _seeded_sql_tool())
    result = node(_state(), {})
    assert result["sql_rows"] == []
    assert result["computed"] == {}
    assert any("evidence unavailable" in note.lower() for note in result["coverage_notes"])
    assert llm.calls == 2
