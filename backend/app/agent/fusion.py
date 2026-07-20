"""Reciprocal rank fusion (RRF): merge several ranked-id lists from
different retrieval systems into a single ordering.

Used by `agent/hybrid_tool.py` to combine VectorTool's dense (cosine
similarity) ranking with TextSearchTool's lexical (Postgres full-text)
ranking -- the two scores aren't on comparable scales (cosine similarity
vs. `ts_rank_cd`), so RRF fuses by *rank position* instead: score(d) = sum
over systems s of 1 / (k + rank_s(d)). A document that ranks well in either
system scores well overall; one that appears in only one system still
contributes via that system's rank alone. k=60 is the constant from the
original Cormack et al. RRF paper -- it damps the difference between
adjacent ranks without needing per-corpus tuning.
"""

DEFAULT_K = 60


def reciprocal_rank_fusion(*ranked_id_lists: list[str], k: int = DEFAULT_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ids in ranked_id_lists:
        for rank, id_ in enumerate(ids, start=1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank)
    return scores
