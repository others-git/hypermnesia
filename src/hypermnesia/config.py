from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class Principal:
    """An authenticated caller and the memory scopes it may read/write."""

    id: str
    scopes: tuple[str, ...] = field(default_factory=tuple)

    def may_access(self, scope: str) -> bool:
        return scope in self.scopes


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HM_", env_file=".env", extra="ignore")

    # --- storage ---
    database_url: str = "postgresql://hypermnesia:hypermnesia@localhost:5432/hypermnesia"

    # --- embeddings (all local / CPU friendly) ---
    embedding_provider: str = "fastembed"  # fastembed | sentence_transformers | ollama
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int | None = None  # auto-detected from the model when None
    ollama_base_url: str = "http://localhost:11434"

    # --- recall / write behaviour ---
    dedupe_threshold: float = 0.92  # cosine sim above which save() updates the near-duplicate
    default_top_k: int = 8

    # Drop hits below this cosine similarity so weak matches don't pollute recall.
    # Tuned for bge-small-en-v1.5, whose cosine range is compressed: unrelated text
    # sits ~0.30-0.45 and relevant hits ~0.55+, so 0.4 trims clear noise while
    # keeping loosely-related memories (missing recall is worse than a weak hit).
    # Re-tune if you change embedding models. 0.0 disables; callers override per-search.
    search_min_similarity: float = 0.4

    # Final ranking blends semantic similarity with recency and importance
    # (generative-agents style): score = w_sim*sim + w_recency*recency + w_importance*imp.
    # similarity dominates by default so relevance still leads.
    score_weight_similarity: float = 1.0
    score_weight_recency: float = 0.25
    score_weight_importance: float = 0.15
    recency_half_life_days: float = 30.0  # last_accessed_at decay half-life
    importance_cap: float = 2.0  # importance is normalised to [0,1] against this cap
    # Candidates fetched by vector distance before re-ranking = k * this multiplier.
    rerank_candidate_multiplier: int = 5

    # Hybrid search: fuse semantic (vector) with lexical (Postgres full-text) recall
    # via reciprocal-rank fusion, so exact tokens (error codes, flag names, paths)
    # aren't lost to the embedding. The fused relevance then feeds the recency/
    # importance blend above. A pure-lexical hit bypasses the similarity floor.
    hybrid_search: bool = True
    rrf_k: int = 60  # reciprocal-rank-fusion constant; larger = flatter rank weighting
    hybrid_vector_weight: float = 1.0
    hybrid_lexical_weight: float = 1.0

    # --- forgetting ---
    # Defaults for the memory_forget sweep: a memory is eligible to be archived
    # once it hasn't been recalled in this many days AND its importance is at or
    # below the floor. Recall and a higher importance both keep a memory alive.
    forget_after_days: float = 180.0
    forget_importance_floor: float = 1.0

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8765

    # --- auth ---
    # JSON: {"<bearer-token>": {"principal": "agent-a", "scopes": ["shared", "user:dev-test"]}}
    auth_tokens: str = "{}"
    require_auth: bool = True

    def principals(self) -> dict[str, Principal]:
        raw = json.loads(self.auth_tokens or "{}")
        out: dict[str, Principal] = {}
        for token, spec in raw.items():
            out[token] = Principal(
                id=str(spec["principal"]),
                scopes=tuple(spec.get("scopes", [])),
            )
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
