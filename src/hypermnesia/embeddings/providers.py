from __future__ import annotations

import json
import urllib.request
from typing import Sequence

from .base import register


@register("fastembed")
class FastEmbedEmbedder:
    """Default backend. ONNX runtime, no PyTorch, runs well on CPU."""

    def __init__(self, model: str, dim: int | None = None, **_):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self.model_id = model
        detected = len(next(iter(self._model.embed(["dimension probe"]))))
        if dim is not None and dim != detected:
            raise ValueError(
                f"Configured dim {dim} != model dim {detected} for {model!r}."
            )
        self.dim = dim or detected

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()


@register("sentence_transformers")
class SentenceTransformersEmbedder:
    """Optional backend (pip install 'hypermnesia[sentence-transformers]')."""

    def __init__(self, model: str, dim: int | None = None, **_):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model, device="cpu")
        self.model_id = model
        detected = int(self._model.get_sentence_embedding_dimension())
        if dim is not None and dim != detected:
            raise ValueError(
                f"Configured dim {dim} != model dim {detected} for {model!r}."
            )
        self.dim = dim or detected

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._model.encode(
            list(texts), normalize_embeddings=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


@register("ollama")
class OllamaEmbedder:
    """Talks to a local Ollama server. Stdlib only (urllib)."""

    def __init__(
        self,
        model: str,
        dim: int | None = None,
        base_url: str = "http://localhost:11434",
        **_,
    ):
        self._url = base_url.rstrip("/") + "/api/embeddings"
        self.model_id = model
        detected = len(self._embed_one("dimension probe"))
        if dim is not None and dim != detected:
            raise ValueError(
                f"Configured dim {dim} != model dim {detected} for {model!r}."
            )
        self.dim = dim or detected

    def _embed_one(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model_id, "prompt": text}).encode()
        req = urllib.request.Request(
            self._url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["embedding"]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)
