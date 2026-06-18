from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool


async def _configure(conn) -> None:
    await register_vector_async(conn)


async def _ensure_extension(database_url: str) -> None:
    """Create the pgvector extension on a raw connection.

    Must run before the pool opens, because the pool's configure callback
    registers the ``vector`` type and fails if the extension doesn't exist yet.
    """
    conn = await psycopg.AsyncConnection.connect(database_url, autocommit=True)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        await conn.close()


async def make_pool(database_url: str) -> AsyncConnectionPool:
    await _ensure_extension(database_url)
    pool = AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        open=False,
    )
    await pool.open()
    return pool


async def init_schema(pool: AsyncConnectionPool, dim: int, model_id: str) -> None:
    """Create the schema and pin the embedding dimension.

    The vector column dimension is fixed once. Swapping to a model with a
    different dimension is refused so search can't silently break.
    """
    async with pool.connection() as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS hm_meta (key text PRIMARY KEY, value text NOT NULL)"
        )

        meta = dict(
            await (
                await conn.execute(
                    "SELECT key, value FROM hm_meta WHERE key IN ('embedding_dim', 'model_id')"
                )
            ).fetchall()
        )
        if meta.get("embedding_dim") and int(meta["embedding_dim"]) != dim:
            raise RuntimeError(
                f"Existing store uses embedding dim {meta['embedding_dim']} but the configured "
                f"model {model_id!r} produces dim {dim}. Re-index (dump, drop, reload) before "
                "switching embedding models."
            )
        if meta.get("model_id") and meta["model_id"] != model_id:
            raise RuntimeError(
                f"Existing store was indexed with model {meta['model_id']!r}; vectors from "
                f"{model_id!r} are not comparable. Re-index before switching embedding models."
            )

        await conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS memories (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                owner_id text NOT NULL,
                scope text NOT NULL,
                type text NOT NULL DEFAULT 'fact',
                content text NOT NULL,
                description text NOT NULL,
                tags text[] NOT NULL DEFAULT '{{}}',
                metadata jsonb NOT NULL DEFAULT '{{}}',
                embedding vector({dim}) NOT NULL,
                model_id text NOT NULL,
                importance real NOT NULL DEFAULT 1.0,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                last_accessed_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # Lexical search column for hybrid (keyword + vector) recall. Generated so
        # it stays in sync with description/content; ADD COLUMN IF NOT EXISTS
        # backfills existing stores and is a no-op once present.
        await conn.execute(
            """
            ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_tsv tsvector
            GENERATED ALWAYS AS (
                to_tsvector('english', coalesce(description, '') || ' ' || coalesce(content, ''))
            ) STORED
            """
        )
        await conn.execute("CREATE INDEX IF NOT EXISTS memories_scope_idx ON memories (scope)")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS memories_tags_idx ON memories USING gin (tags)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS memories_tsv_idx ON memories USING gin (content_tsv)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS memories_embedding_idx "
            "ON memories USING hnsw (embedding vector_cosine_ops)"
        )
        await conn.execute(
            """
            INSERT INTO hm_meta (key, value) VALUES ('embedding_dim', %s), ('model_id', %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            (str(dim), model_id),
        )
