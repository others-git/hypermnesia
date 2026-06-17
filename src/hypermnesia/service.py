from __future__ import annotations

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
    ) -> tuple[Memory, bool]:
        """Insert a memory, or update the nearest duplicate within the same scope.

        Returns (memory, created) where created is False when an existing
        near-duplicate was updated instead.
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
                return Memory(**row), False

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
            return Memory(**row), True

    async def search(
        self,
        *,
        query: str,
        scopes: list[str],
        tags: list[str] | None = None,
        k: int = 8,
    ) -> list[SearchHit]:
        if not scopes:
            return []
        vec = Vector(self.embedder.embed_query(query))
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = await (
                await conn.execute(
                    f"""
                    SELECT {_SELECT_COLS}, 1 - (embedding <=> %(q)s) AS similarity
                    FROM memories
                    WHERE scope = ANY(%(scopes)s)
                      AND (%(tags)s::text[] IS NULL OR tags && %(tags)s)
                    ORDER BY embedding <=> %(q)s
                    LIMIT %(k)s
                    """,
                    {"q": vec, "scopes": scopes, "tags": tags, "k": k},
                )
            ).fetchall()
            if rows:
                await conn.execute(
                    "UPDATE memories SET last_accessed_at = now() WHERE id = ANY(%s)",
                    ([r["id"] for r in rows],),
                )
        return [SearchHit(**r) for r in rows]

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
