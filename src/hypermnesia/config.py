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
    # Second dedup gate: a merge additionally requires the CONTENTS to be this
    # similar. The combined description+content embedding can clear
    # dedupe_threshold on the description alone, and merging two different
    # facts that merely share a description shape silently destroys one of
    # them. Failing the gate inserts instead (duplicates are recoverable;
    # clobbered facts are not). 0 disables.
    dedupe_content_threshold: float = 0.9
    default_top_k: int = 8

    # Drop hits below this cosine similarity so weak matches don't pollute recall.
    # Tuned for bge-small-en-v1.5, whose cosine range is compressed: unrelated text
    # sits ~0.30-0.45 and relevant hits ~0.55+, so 0.4 trims clear noise while
    # keeping loosely-related memories (missing recall is worse than a weak hit).
    # Re-tune if you change embedding models. 0.0 disables; callers override per-search.
    search_min_similarity: float = 0.4

    # Second recall gate, relative instead of absolute: drop a hit whose cosine
    # similarity falls more than this far below the best hit of the same search.
    # An absolute floor cannot separate signal from noise on its own, because
    # every model has its own cosine range and unrelated text still scores well
    # inside it (measured on bge-small: unrelated memories sit ~0.43-0.62 while
    # relevant ones sit ~0.50-0.85, so the two overlap and *no* single floor
    # splits them). The distance to the best hit does separate them, and it
    # travels across models: on an 11-query memory-shaped eval this kept every
    # relevant hit while cutting irrelevant ones from 77 to 15, where the
    # tightest absolute floor that trimmed as much already lost a relevant hit.
    # A flat, standout-free result set (an off-topic query) is left alone, so
    # this narrows a good answer's neighbourhood rather than inventing one.
    # Lexical hits bypass it, as they do the floor. 0.0 disables; callers
    # override per-search.
    search_relative_cutoff: float = 0.15

    # Final ranking blends semantic similarity with recency and importance
    # (generative-agents style): score = w_sim*sim + w_recency*recency + w_importance*imp.
    #
    # These weights must be read against how much relevance actually VARIES, not
    # against its nominal [0,1] range. Embedding models compress cosine into a
    # narrow band — for bge-small-en-v1.5 every plausible hit lands in roughly
    # 0.55-0.73 — so relevance moves a score by ~0.18 across a candidate set.
    # The original 0.25/0.15 pair therefore let recency alone outweigh the entire
    # usable similarity spread, and a stale-but-important memory beat the one
    # that answered the query, despite the comment here claiming similarity led.
    # Damped to a fraction of that spread so recency and importance break ties
    # within a neighbourhood, which is what they are for, rather than choosing it.
    score_weight_similarity: float = 1.0
    score_weight_recency: float = 0.10
    score_weight_importance: float = 0.06
    recency_half_life_days: float = 30.0  # last_accessed_at decay half-life
    importance_cap: float = 2.0  # importance is normalised to [0,1] against this cap
    # Candidates fetched by vector distance before re-ranking = k * this multiplier.
    rerank_candidate_multiplier: int = 5

    # Hybrid search: combine semantic (vector) with lexical (Postgres full-text)
    # recall so exact tokens (error codes, flag names, paths) aren't lost to the
    # embedding. Relevance is the cosine similarity itself, plus a bounded bonus
    # for appearing in the lexical list; a pure-lexical hit also bypasses the
    # similarity floor.
    #
    # This deliberately is NOT reciprocal-rank fusion. RRF keeps only ranks, so
    # normalising it against "rank 1 in both lists" gave every lexical match a
    # flat ~0.5 relevance jump over a vector-only hit — far more than the whole
    # usable similarity spread, so a memory that merely happened to contain the
    # query's function words outranked the one that answered it. Because
    # websearch_to_tsquery ANDs every term, which memories match lexically is
    # near-arbitrary for conversational queries, so that jump was mostly noise.
    # Keeping calibrated similarity as the base and capping the lexical bonus
    # preserves exact-token recall without letting it overrule meaning.
    hybrid_search: bool = True
    # Most a lexical match can add to relevance, at lexical rank 1; the bonus
    # decays with rank. Measured: 0.05-0.30 all behave the same on a labelled
    # recall set, so this is a plateau rather than a knife-edge tuning constant.
    hybrid_lexical_boost: float = 0.10
    # Rank-decay constant for that bonus; larger = flatter across lexical ranks.
    hybrid_rank_decay_k: int = 60

    # --- observability ---
    # Log every memory_search (query, candidate/hit counts, top scores, latency)
    # to the search_log table; memory_stats aggregates it. Cheap (one insert per
    # search) but queries land in the DB — disable if that's unwanted.
    search_log_enabled: bool = True

    # --- forgetting ---
    # Defaults for the memory_forget sweep: a memory is eligible to be archived
    # once it hasn't been recalled in this many days AND its importance is at or
    # below the floor. Recall and a higher importance both keep a memory alive.
    forget_after_days: float = 180.0
    forget_importance_floor: float = 1.0
    # Opt-in periodic sweep: every this many hours the server archives what
    # memory_forget(apply=true) would, over every scope, using the two
    # thresholds above. 0 (the default) disables it — forgetting then only
    # happens when a client calls memory_forget explicitly.
    forget_sweep_hours: float = 0.0

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8765
    # Root log level for the `hypermnesia` logger. Without this nothing
    # configures logging, so the forget sweep's report of what it archived —
    # and the debug line for a failed roots handshake — go nowhere.
    log_level: str = "INFO"

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
