"""TextSearchTool tests -- offline, stubbing the SQLAlchemy engine/connection
so no real Postgres (or its full-text functions) is needed."""

from app.agent.text_search_tool import TextSearchTool


class _StubResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _StubConn:
    def __init__(self, rows_by_company, *, raise_on_execute=False):
        self._rows_by_company = rows_by_company
        self._raise = raise_on_execute
        self.calls = []

    def execute(self, stmt, params):
        if self._raise:
            raise RuntimeError("relation \"chunk_text\" does not exist")
        self.calls.append(params)
        return _StubResult(self._rows_by_company.get(params["company"], []))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _StubEngine:
    def __init__(self, rows_by_company, *, raise_on_execute=False):
        self._conn = _StubConn(rows_by_company, raise_on_execute=raise_on_execute)

    def connect(self):
        return self._conn


def _row(id_, company, rank, text="some 10-K text", source="Meta_10K.pdf", page=5):
    return {"id": id_, "company": company, "source": source, "page": page, "text": text, "rank": rank}


def test_queries_once_per_company():
    engine = _StubEngine({"Apple": [], "Meta": []})
    TextSearchTool(engine).query("Item 1A risk factors", ["Apple", "Meta"])
    assert [c["company"] for c in engine._conn.calls] == ["Apple", "Meta"]


def test_no_companies_skips_query_entirely():
    engine = _StubEngine({})
    hits = TextSearchTool(engine).query("anything", [])
    assert hits == []
    assert engine._conn.calls == []


def test_returns_hits_as_typed_objects():
    engine = _StubEngine({"Apple": [_row("c1", "Apple", 0.7)]})
    hits = TextSearchTool(engine).query("Item 1A", ["Apple"])
    assert len(hits) == 1
    assert hits[0].id == "c1"
    assert hits[0].company == "Apple"
    assert hits[0].rank == 0.7


def test_top_k_is_forwarded_as_a_query_param():
    engine = _StubEngine({"Apple": []})
    TextSearchTool(engine, top_k=3).query("revenue growth", ["Apple"])
    assert engine._conn.calls[0]["top_k"] == 3


def test_db_error_degrades_to_no_hits_instead_of_raising():
    engine = _StubEngine({}, raise_on_execute=True)
    hits = TextSearchTool(engine).query("revenue growth", ["Apple"])
    assert hits == []


def test_question_is_tokenized_into_an_or_combined_tsquery():
    # plainto_tsquery/websearch_to_tsquery AND every word together, which
    # almost never matches a real chunk for a multi-word natural-language
    # question (verified live) -- the tsquery sent to Postgres must instead
    # OR the question's words so a partial match still surfaces.
    engine = _StubEngine({"Meta": []})
    TextSearchTool(engine).query("What does Item 1A say about competition?", ["Meta"])
    tsq = engine._conn.calls[0]["tsq"]
    assert " | " in tsq
    assert "item" in tsq
    assert "1a" in tsq
    assert "competition" in tsq


def test_punctuation_only_question_skips_the_query_entirely():
    engine = _StubEngine({"Meta": []})
    hits = TextSearchTool(engine).query("???", ["Meta"])
    assert hits == []
    assert engine._conn.calls == []
