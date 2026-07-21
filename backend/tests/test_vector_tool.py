"""VectorTool tests -- offline, stubbing both the embedder and the
Pinecone index so no network/API call happens.
"""

from app.agent.vector_tool import VectorTool


class _StubEmbedder:
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _StubIndex:
    """Records one query call per company and returns pre-scripted matches."""

    def __init__(self, matches_by_company):
        self._matches_by_company = matches_by_company
        self.calls = []

    def query(self, *, vector, top_k, filter, include_metadata):
        company = filter["company"]["$eq"]
        self.calls.append(company)
        return {"matches": self._matches_by_company.get(company, [])}


def _match(id_, score, **metadata):
    return {"id": id_, "score": score, "metadata": metadata}


def test_queries_once_per_company():
    index = _StubIndex({"Apple": [], "Meta": []})
    VectorTool(index, _StubEmbedder()).query("why did revenue grow?", ["Apple", "Meta"])
    assert index.calls == ["Apple", "Meta"]


def test_no_companies_skips_query_entirely():
    index = _StubIndex({})
    result = VectorTool(index, _StubEmbedder()).query("anything", [])
    assert result.chunks == []
    assert result.rejected == []
    assert index.calls == []


def test_above_floor_chunks_are_kept_with_metadata():
    index = _StubIndex(
        {"Apple": [_match("c1", 0.8, text="Apple grew because...", source="Apple_10K.pdf", page=12)]}
    )
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple"])
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.id == "c1"
    assert chunk.company == "Apple"
    assert chunk.score == 0.8
    assert chunk.source == "Apple_10K.pdf"
    assert chunk.page == 12
    assert chunk.text == "Apple grew because..."
    assert result.rejected == []


def test_below_floor_chunks_are_rejected_not_dropped_silently():
    index = _StubIndex({"Apple": [_match("c2", 0.1, text="boilerplate header noise")]})
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple"])
    assert result.chunks == []
    assert len(result.rejected) == 1
    assert result.rejected[0].id == "c2"
    assert result.rejected[0].company == "Apple"
    assert result.rejected[0].score == 0.1
    assert result.rejected[0].reason == "below_floor"


def test_print_to_pdf_header_noise_is_rejected_as_boilerplate_even_above_floor():
    noisy_text = "4/20/26, 12:05 PM goog-20251231 file:///Users/x/Downloads/goog-20251231.htm"
    index = _StubIndex({"Apple": [_match("c3", 0.6, text=noisy_text)]})
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple"])
    assert result.chunks == []
    assert len(result.rejected) == 1
    assert result.rejected[0].id == "c3"
    assert result.rejected[0].score == 0.6
    assert result.rejected[0].reason == "boilerplate"


def test_duplicate_text_within_a_company_keeps_only_the_highest_scoring_copy():
    text = "Our revenue grew due to strength in Search and Cloud advertising demand."
    index = _StubIndex(
        {"Apple": [_match("dup-lo", 0.5, text=text), _match("dup-hi", 0.7, text=text)]}
    )
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple"])
    assert len(result.chunks) == 1
    assert result.chunks[0].id == "dup-hi"
    assert result.chunks[0].score == 0.7
    assert len(result.rejected) == 1
    assert result.rejected[0].id == "dup-lo"
    assert result.rejected[0].reason == "duplicate"


def test_duplicate_text_across_different_companies_is_not_deduped():
    text = "Identical boilerplate legal text that happens to appear in both filings."
    index = _StubIndex(
        {"Apple": [_match("a1", 0.6, text=text)], "Meta": [_match("m1", 0.6, text=text)]}
    )
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple", "Meta"])
    assert len(result.chunks) == 2
    assert result.rejected == []


def test_equal_score_duplicates_keep_the_first_seen_copy():
    text = "Some substantive strategy narrative here about revenue drivers."
    index = _StubIndex(
        {"Apple": [_match("first", 0.5, text=text), _match("second", 0.5, text=text)]}
    )
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple"])
    assert len(result.chunks) == 1
    assert result.chunks[0].id == "first"


def test_substantive_text_with_incidental_header_line_is_kept():
    text = (
        "4/20/26, 12:05 PM goog-20251231 file:///Users/x/goog.htm\n"
        "Our revenue grew due to strength in Search and Cloud, driven by "
        "increased advertiser demand and enterprise adoption of AI products."
    )
    index = _StubIndex({"Apple": [_match("c4", 0.6, text=text)]})
    result = VectorTool(index, _StubEmbedder(), score_floor=0.25).query("why", ["Apple"])
    assert len(result.chunks) == 1
    assert result.chunks[0].id == "c4"
    assert result.rejected == []
