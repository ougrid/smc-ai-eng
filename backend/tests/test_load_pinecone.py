"""Tests for scripts/load_pinecone.py's pure metadata-normalization logic —
no Pinecone/network needed. Canonical company mapping is deliberately verified
here: Alphabet_10K_FY2025.pdf -> "Google" (matches the SQL table's company
names, not the PDF filename)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

_spec = importlib.util.spec_from_file_location(
    "load_pinecone", REPO_ROOT / "scripts" / "load_pinecone.py"
)
load_pinecone = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(load_pinecone)


def _record(source_basename: str) -> dict:
    return {
        "id": "abc123",
        "namespace": "__default__",
        "values": [0.1, 0.2],
        "metadata": {
            "creationdate": "2026-04-20T05:05:41+00:00",
            "creator": "Mozilla/5.0",
            "moddate": "2026-04-20T05:05:41+00:00",
            "producer": "Skia/PDF m147",
            "page": 66,
            "page_label": "67",
            "source": f"/Users/someone/projects/smc/10k_filings/{source_basename}",
            "title": "goog-20251231",
            "total_pages": 156,
            "text": "some chunk text",
        },
    }


@pytest.mark.parametrize(
    "filename,expected_company",
    [
        ("Alphabet_10K_FY2025.pdf", "Google"),
        ("Amazon_10K_FY2025.pdf", "Amazon"),
        ("Apple_10K_FY2025.pdf", "Apple"),
        ("Meta_10K_FY2025.pdf", "Meta"),
    ],
)
def test_normalize_record_maps_canonical_company(filename, expected_company):
    normalized = load_pinecone.normalize_record(_record(filename))
    assert normalized["metadata"]["company"] == expected_company
    assert normalized["metadata"]["source"] == filename


def test_normalize_record_drops_junk_keys():
    normalized = load_pinecone.normalize_record(_record("Meta_10K_FY2025.pdf"))
    for key in ("creator", "producer", "moddate", "creationdate"):
        assert key not in normalized["metadata"]
    assert normalized["metadata"]["text"] == "some chunk text"
    assert normalized["metadata"]["page"] == 66


def test_normalize_record_fails_loudly_on_unknown_file():
    with pytest.raises(KeyError):
        load_pinecone.normalize_record(_record("Microsoft_10K_FY2025.pdf"))


def test_iter_batches_batches_correctly(tmp_path):
    import gzip
    import json

    records = [_record("Apple_10K_FY2025.pdf") | {"id": f"id-{i}"} for i in range(5)]
    path = tmp_path / "vectors.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    batches = list(load_pinecone.iter_batches(path, batch_size=2))
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sum(len(b) for b in batches) == 5
