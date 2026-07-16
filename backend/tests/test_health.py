"""Offline test of the create_app() DI factory: injected stub engine/pinecone
index, no real Postgres/Pinecone/OpenAI needed — proves the health endpoint
contract without touching the network."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


class _StubResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class _StubConn:
    def __init__(self, row_count):
        self._row_count = row_count

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _query):
        return _StubResult(self._row_count)


class _StubEngine:
    def __init__(self, row_count):
        self._row_count = row_count

    def connect(self):
        return _StubConn(self._row_count)


class _StubPineconeIndex:
    def __init__(self, count):
        self._count = count

    def describe_index_stats(self):
        return type("Stats", (), {"total_vector_count": self._count})()


def _make_client(*, postgres_rows: int, vector_count: int) -> TestClient:
    settings = Settings(openai_api_key="test")
    app = create_app(
        settings=settings,
        engine=_StubEngine(postgres_rows),
        agent_engine=_StubEngine(postgres_rows),
        pinecone_index=_StubPineconeIndex(vector_count),
    )
    return TestClient(app)


def test_health_ok_when_counts_match():
    with _make_client(postgres_rows=192, vector_count=4072) as client:
        resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"postgres_rows": 192, "vector_count": 4072, "ok": True}


def test_health_not_ok_when_vector_count_mismatches():
    with _make_client(postgres_rows=192, vector_count=0) as client:
        resp = client.get("/api/health")
    assert resp.json() == {"postgres_rows": 192, "vector_count": 0, "ok": False}


def test_health_not_ok_when_postgres_rows_mismatch():
    with _make_client(postgres_rows=48, vector_count=4072) as client:
        resp = client.get("/api/health")
    assert resp.json() == {"postgres_rows": 48, "vector_count": 4072, "ok": False}
