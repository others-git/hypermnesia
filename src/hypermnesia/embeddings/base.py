from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Pluggable embedding backend.

    Implementations must set ``model_id`` and ``dim`` after construction and
    return L2-comparable vectors (cosine distance is used downstream, so
    normalization is optional but recommended).
    """

    model_id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return deco


def create_embedder(provider: str, model: str, dim: int | None = None, **kwargs) -> Embedder:
    # Import for side effects so providers register themselves.
    from . import providers  # noqa: F401

    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown embedding provider {provider!r}. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[provider](model=model, dim=dim, **kwargs)
