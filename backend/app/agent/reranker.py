"""Cross-encoder scoring: the model half of reranking (see
agent/reranked_tool.py for the per-company cut/cost logic that consumes it).

A cross-encoder scores a (question, chunk_text) pair jointly -- unlike the
dense/lexical retrieval that came before it, which score each chunk against
the question independently -- so it's the highest-precision signal in this
pipeline, at the cost of being too slow to run over every chunk in the
store. That's why it only ever sees the already-narrowed hybrid pool
(agent/hybrid_tool.py), never the raw index.

`CrossEncoder` is imported at module level (not lazily inside `__init__`)
specifically so tests can monkeypatch `app.agent.reranker.CrossEncoder`
with a stub -- the real model (BAAI/bge-reranker-v2-m3, ~1GB) is downloaded
from Hugging Face on first real construction, which only ever happens in
main.py's `_build_real_graph`, never in an offline test.
"""

from typing import Protocol

from sentence_transformers import CrossEncoder


class Reranker(Protocol):
    def score(self, question: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self._model = CrossEncoder(model_name)

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(question, text) for text in texts]
        return [float(s) for s in self._model.predict(pairs)]
