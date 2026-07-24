"""Unit tests for the content dedup gate (no DB).

The first dedup gate compares one combined description+content embedding, so
two different facts sharing a description shape can look like duplicates and
one would silently clobber the other. The second gate compares the contents
directly; these tests pin its decision logic.
"""

from __future__ import annotations

import pytest

from hypermnesia.config import Settings
from hypermnesia.service import MemoryService


class VecEmbedder:
    """Fake embedder returning fixed vectors per text, counting calls."""

    model_id = "vec"
    dim = 2

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [self.table[t] for t in texts]

    def embed_query(self, text):
        return self.table[text]


def _svc(embedder=None, **overrides) -> MemoryService:
    return MemoryService(pool=None, embedder=embedder, settings=Settings(**overrides))


async def test_identical_content_matches_without_embedding():
    emb = VecEmbedder({})
    assert await _svc(emb)._contents_match("same text", "same text") is True
    assert emb.calls == 0


async def test_disabled_gate_always_matches():
    emb = VecEmbedder({})
    svc = _svc(emb, dedupe_content_threshold=0.0)
    assert await svc._contents_match("one fact", "a different fact") is True
    assert emb.calls == 0


async def test_orthogonal_contents_do_not_match():
    emb = VecEmbedder({"fact a": [1.0, 0.0], "fact b": [0.0, 1.0]})
    assert await _svc(emb)._contents_match("fact a", "fact b") is False


async def test_near_identical_contents_match():
    emb = VecEmbedder({"fact a": [1.0, 0.0], "fact a2": [0.99, 0.05]})
    assert await _svc(emb)._contents_match("fact a", "fact a2") is True


async def test_zero_vector_never_matches():
    emb = VecEmbedder({"fact a": [0.0, 0.0], "fact b": [0.0, 1.0]})
    assert await _svc(emb)._contents_match("fact a", "fact b") is False


@pytest.mark.parametrize("threshold", [0.5, 0.9])
async def test_threshold_is_respected(threshold):
    # cosine between these is exactly ~0.7071
    emb = VecEmbedder({"a": [1.0, 0.0], "b": [1.0, 1.0]})
    svc = _svc(emb, dedupe_content_threshold=threshold)
    assert (await svc._contents_match("a", "b")) is (threshold <= 0.7071)
