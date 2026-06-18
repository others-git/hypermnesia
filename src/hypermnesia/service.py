from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pgvector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from .config import Settings
from .db import init_schema, make_pool
from .embeddings import Embedder, create_embedder
from .models import Memory, SearchHit

_SELECT_COLS = (
    "id::text, owner_id, scope, type, content, description, tags, metadata, "
    "importance, created_at, updated_at, last_accessed_at"
)


class MemoryService:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, settings: Settings):
        self.pool = pool
        self.embedder = embedder
        self.settings = settings

    @classmethod
    async def create(cls, settings: Settings) -> "MemoryService":
        embedder = create_embedder(
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_dim,
            base_url=settings.ollama_base_url,
        )
        pool = await make_pool(settings.database_url)
        await init_schema(pool, embedder.dim, embedder.model_id)
        return cls(pool, embedder, settings)

    async def aclose(self) -> None:
        await self.pool.close()

    def _embed_record(self, description: str, content: str) -> Vector:
        return Vector(self.embedder.embed_documents([f"{description}\n\n{content}"])[0])

    async def save(
        self,
        *,
        owner_id: str,
        scope: str,
        content: str,
        description: str,
        type: str = "fact",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float = 1.0,
    ) -> tuple[Memory, bool, Memory | None]:
        """Insert a memory, or update the nearest duplicate within the same scope.

        Returns (memory, created, replaced). ``created`` is False when an existing
        near-duplicate was updated instead of inserted; ``replaced`` is then the
        memory as it looked *before* the merge (None on insert), so callers can
        see what an overwrite clobbered and catch a bad merge.
        """
        tags = tags or []
        metadata = metadata or {}
        vec = self._embed_record(description, content)

        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            dup = await (
                await conn.execute(
                    f"""
                    SELECT {_SELECT_COLS}, 1 - (embedding <=> %s) AS similarity
                    FROM memories
                    WHERE scope = %s
                    ORDER BY embedding <=> %s
                    LIMIT 1
                    """,
                    (vec, scope, vec),
                )
            ).fetchone()

            if dup and dup["similarity"] >= self.settings.dedupe_threshold:
                replaced = Memory(**{k: v for k, v in dup.items() if k != "similarity"})
                row = await (
                    await conn.execute(
                        f"""
                        UPDATE memories
                        SET content = %s, description = %s, type = %s, tags = %s,
                            metadata = %s, importance = %s, embedding = %s,
                            model_id = %s, updated_at = now(), last_accessed_at = now()
                        WHERE id = %s
                        RETURNING {_SELECT_COLS}
                        """,
                        (
                            content, description, type, tags, Jsonb(metadata),
                            importance, vec, self.embedder.model_id, dup["id"],
                        ),
                    )
                ).fetchone()
                return Memory(**row), False, replaced

            row = await (
                await conn.execute(
                    f"""
                    INSERT INTO memories
                        (owner_id, scope, type, content, description, tags,
                         metadata, embedding, model_id, importance)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING {_SELECT_COLS}
                    """,
                    (
                        owner_id, scope, type, content, description, tags,
                        Jsonb(metadata), vec, self.embedder.model_id, importance,
                    ),
                )
            ).fetchone()
            return Memory(**row), True, None

    def _rank_score(self, similarity: float, importance: float, last_accessed: datetime | None) -> float:
        """Blend semantic similarity with recency decay and importance.

        Recency decays exponentially from ``last_accessed_at`` with the configured
        half-life; importance is normalised to [0,1] against ``importance_cap``.
        """
        s = self.settings
        if last_accessed is not None:
            now = datetime.now(last_accessed.tzinfo or timezone.utc)
            age_days = max((now - last_accessed).total_seconds() / 86400.0, 0.0)
            recency = 0.5 ** (age_days / s.recency_half_life_days)
        else:
            recency = 0.0
        imp = min(max(importance, 0.0), s.importance_cap) / s.importance_cap
        return (
            s.score_weight_similarity * similarity
            + s.score_weight_recency * recency
            + s.score_weight_importance * imp
        )

    async def search(
        self,
        *,
        query: str,
        scopes: list[str],
        tags: list[str] | None = None,
        k: int = 8,
        min_similarity: float | None = None,
    ) -> list[SearchHit]:
        if not scopes:
            return []
        floor = (
            self.settings.search_min_similarity
            if min_similarity is None
            else min_similarity
        )
        # Pull a wider candidate pool by vector distance, then re-rank in Python so
        # recency/importance can reorder within the relevant neighbourhood.
        pool = max(k * self.settings.rerank_candidate_multiplier, k)
        vec = Vector(self.embedder.embed_query(query))
        # HNSW post-filters on scope/tags during the index walk, so a filtered
        # search can return fewer than `pool` rows. Widen ef_search past the pool
        # (txn-local) so the scan keeps enough candidates to fill it. Capped at the
        # pgvector max of 1000.
        ef_search = min(max(pool * 2, 40), 1000)
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            rows = await (
                await conn.execute(
                    f"""
                    SELECT {_SELECT_COLS}, 1 - (embedding <=> %(q)s) AS similarity
                    FROM memories
                    WHERE scope = ANY(%(scopes)s)
                      AND (%(tags)s::text[] IS NULL OR tags && %(tags)s)
                    ORDER BY embedding <=> %(q)s
                    LIMIT %(pool)s
                    """,
                    {"q": vec, "scopes": scopes, "tags": tags, "pool": pool},
                )
            ).fetchall()

            hits: list[SearchHit] = []
            for r in rows:
                if r["similarity"] < floor:
                    continue
                hit = SearchHit(**r)
                hit.score = self._rank_score(
                    hit.similarity, hit.importance, hit.last_accessed_at
                )
                hits.append(hit)
            hits.sort(key=lambda h: h.score, reverse=True)
            hits = hits[:k]

            if hits:
                await conn.execute(
                    "UPDATE memories SET last_accessed_at = now() WHERE id = ANY(%s)",
                    ([h.id for h in hits],),
                )
        return hits

    async def get(self, memory_id: str, scopes: list[str]) -> Memory | None:
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            row = await (
                await conn.execute(
                    f"SELECT {_SELECT_COLS} FROM memories "
                    "WHERE id = %s AND scope = ANY(%s)",
                    (memory_id, scopes),
                )
            ).fetchone()
        return Memory(**row) if row else None

    async def update(
        self,
        memory_id: str,
        scopes: list[str],
        *,
        content: str | None = None,
        description: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
    ) -> Memory | None:
        """Edit a memory by id. Only provided fields change; others are kept.

        Re-embeds when content or description changes. Returns the updated memory,
        or None if it doesn't exist in an accessible scope.
        """
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            existing = await (
                await conn.execute(
                    f"SELECT {_SELECT_COLS} FROM memories "
                    "WHERE id = %s AND scope = ANY(%s)",
                    (memory_id, scopes),
                )
            ).fetchone()
            if existing is None:
                return None

            new_content = existing["content"] if content is None else content
            new_description = existing["description"] if description is None else description
            sets = {
                "content": new_content,
                "description": new_description,
                "type": existing["type"] if type is None else type,
                "tags": existing["tags"] if tags is None else tags,
                "metadata": Jsonb(existing["metadata"] if metadata is None else metadata),
                "importance": existing["importance"] if importance is None else importance,
            }
            if content is not None or description is not None:
                sets["embedding"] = self._embed_record(new_description, new_content)
                sets["model_id"] = self.embedder.model_id

            assignments = ", ".join(f"{col} = %s" for col in sets)
            row = await (
                await conn.execute(
                    f"""
                    UPDATE memories SET {assignments}, updated_at = now()
                    WHERE id = %s
                    RETURNING {_SELECT_COLS}
                    """,
                    (*sets.values(), memory_id),
                )
            ).fetchone()
        return Memory(**row)

    async def list(
        self, scopes: list[str], tags: list[str] | None = None, limit: int = 50
    ) -> list[Memory]:
        if not scopes:
            return []
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = await (
                await conn.execute(
                    f"""
                    SELECT {_SELECT_COLS} FROM memories
                    WHERE scope = ANY(%(scopes)s)
                      AND (%(tags)s::text[] IS NULL OR tags && %(tags)s)
                    ORDER BY updated_at DESC
                    LIMIT %(limit)s
                    """,
                    {"scopes": scopes, "tags": tags, "limit": limit},
                )
            ).fetchall()
        return [Memory(**r) for r in rows]

    async def delete(self, memory_id: str, scopes: list[str]) -> bool:
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                "DELETE FROM memories WHERE id = %s AND scope = ANY(%s)",
                (memory_id, scopes),
            )
        return cur.rowcount > 0
