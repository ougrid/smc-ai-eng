"""CrossEncoderReranker tests -- offline. Patches app.agent.reranker.CrossEncoder
with a stub so the real ~1GB model is never downloaded/loaded in tests.
"""

from unittest.mock import patch

from app.agent.reranker import CrossEncoderReranker


class _StubCrossEncoder:
    """Deterministic stand-in: score = text length, so ordering is checkable
    without depending on any real model's actual relevance judgments."""

    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        return [len(text) for _, text in pairs]


def test_score_returns_one_float_per_text_in_order():
    with patch("app.agent.reranker.CrossEncoder", _StubCrossEncoder):
        reranker = CrossEncoderReranker("fake-model")
        scores = reranker.score("q", ["short", "a longer piece of text"])
    assert scores == [5.0, 22.0]
    assert all(isinstance(s, float) for s in scores)


def test_score_pairs_the_question_with_every_text():
    captured = {}

    class _CapturingCrossEncoder:
        def __init__(self, model_name):
            pass

        def predict(self, pairs):
            captured["pairs"] = pairs
            return [0.0 for _ in pairs]

    with patch("app.agent.reranker.CrossEncoder", _CapturingCrossEncoder):
        CrossEncoderReranker("fake-model").score("why did revenue grow?", ["a", "b"])
    assert captured["pairs"] == [("why did revenue grow?", "a"), ("why did revenue grow?", "b")]


def test_score_of_no_texts_returns_empty_without_calling_predict():
    class _RaisingCrossEncoder:
        def __init__(self, model_name):
            pass

        def predict(self, pairs):
            raise AssertionError("predict should not be called for an empty text list")

    with patch("app.agent.reranker.CrossEncoder", _RaisingCrossEncoder):
        assert CrossEncoderReranker("fake-model").score("q", []) == []


def test_model_name_is_passed_through_to_cross_encoder():
    with patch("app.agent.reranker.CrossEncoder", _StubCrossEncoder):
        reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
    assert reranker._model.model_name == "BAAI/bge-reranker-v2-m3"
