"""Reciprocal rank fusion -- pure function, no stubs needed."""

from app.agent.fusion import reciprocal_rank_fusion


def test_single_list_preserves_rank_order():
    scores = reciprocal_rank_fusion(["a", "b", "c"])
    assert scores["a"] > scores["b"] > scores["c"]


def test_id_ranked_well_in_both_lists_beats_one_ranked_well_in_only_one():
    scores = reciprocal_rank_fusion(["a", "b"], ["a", "c"])
    assert scores["a"] > scores["b"]
    assert scores["a"] > scores["c"]


def test_disjoint_lists_still_score_every_id():
    scores = reciprocal_rank_fusion(["a", "b"], ["c", "d"])
    assert set(scores) == {"a", "b", "c", "d"}


def test_empty_lists_yield_no_scores():
    assert reciprocal_rank_fusion([], []) == {}


def test_smaller_k_amplifies_rank_differences():
    loose = reciprocal_rank_fusion(["a", "b"], k=60)
    tight = reciprocal_rank_fusion(["a", "b"], k=1)
    assert (tight["a"] - tight["b"]) > (loose["a"] - loose["b"])
