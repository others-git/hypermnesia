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

    # --- server ---
    host: str = "127.0.0.1"
    port: int = 8000

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
