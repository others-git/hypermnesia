import pytest

from hypermnesia.embeddings import create_embedder, register


@register("fake")
class _FakeEmbedder:
    def __init__(self, model, dim=None, **_):
        self.model_id = model
        self.dim = dim or 3

    def embed_documents(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0, 1.0]


def test_create_known_provider():
    emb = create_embedder("fake", "m", dim=3)
    assert emb.model_id == "m"
    assert emb.dim == 3
    assert emb.embed_query("hi") == [2.0, 0.0, 1.0]


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        create_embedder("does-not-exist", "m")
