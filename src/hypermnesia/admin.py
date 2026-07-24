"""Admin operations for the `hypermnesia` CLI: export, import, reindex.

These run on the server host with direct database access — they bypass MCP
auth and scopes on purpose (they are the backup and migration story, not agent
tools). Export writes JSONL without embeddings; import and reindex re-embed
with the configured model, so a dump survives an embedding-model change.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncIterator, TextIO

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import Settings
from .db import init_schema, make_pool
from .embeddings import create_embedder, record_text

EXPORT_FIELDS = (
    "id", "owner_id", "scope", "type", "content", "description",
    "tags", "metadata", "importance",
    "created_at", "updated_at", "last_accessed_at", "archived_at",
)
_TS_FIELDS = ("created_at", "updated_at", "last_accessed_at", "archived_at")


def row_to_line(row: dict[str, Any]) -> str:
    """Serialize one memory row to a JSONL line (timestamps as ISO-8601)."""
    out: dict[str, Any] = {}
    for f in EXPORT_FIELDS:
        v = row[f]
        out[f] = v.isoformat() if f in _TS_FIELDS and v is not None else v
    return json.dumps(out, ensure_ascii=False)


def line_to_row(line: str) -> dict[str, Any]:
    """Parse a JSONL line back into a row dict with real datetimes."""
    row = json.loads(line)
    missing = [f for f in EXPORT_FIELDS if f not in row]
    if missing:
        raise ValueError(f"export line is missing fields: {missing}")
    for f in _TS_FIELDS:
        if row[f] is not None:
            row[f] = datetime.fromisoformat(row[f])
    return row


async def export_memories(
    database_url: str, out: TextIO, scopes: list[str] | None = None
) -> int:
    """Dump memories (including archived, embeddings excluded) as JSONL."""
    cols = "id::text, " + ", ".join(f for f in EXPORT_FIELDS if f != "id")
    where = " WHERE scope = ANY(%s)" if scopes else ""
    params = (scopes,) if scopes else ()
    count = 0
    async with await psycopg.AsyncConnection.connect(database_url) as conn:
        # Server-side cursor: stream instead of loading the whole store.
        async with conn.cursor(name="hm_export", row_factory=dict_row) as cur:
            await cur.execute(
                f"SELECT {cols} FROM memories{where} ORDER BY created_at", params
            )
            async for row in cur:
                out.write(row_to_line(row) + "\n")
                count += 1
    return count


def _read_dump(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(line_to_row(line))
            except (ValueError, KeyError, json.JSONDecodeError) as e:
                raise ValueError(f"{path}:{n}: bad export line: {e}") from e
    return rows


async def import_memories(
    settings: Settings, path: str, batch_size: int = 64
) -> dict[str, int]:
    """Restore a JSONL dump, re-embedding with the configured model.

    Ids and timestamps are preserved; rows whose id already exists are skipped
    (an import never overwrites), so re-running a restore is idempotent.
    """
    rows = _read_dump(path)
    embedder = create_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dim,
        base_url=settings.ollama_base_url,
    )
    pool = await make_pool(settings.database_url)
    try:
        # Creates the schema on a fresh database, and refuses a store whose
        # pinned model/dim doesn't match the configured embedder.
        await init_schema(pool, embedder.dim, embedder.model_id)
        inserted = 0
        async with pool.connection() as conn:
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                vecs = embedder.embed_documents(
                    [record_text(r["description"], r["content"]) for r in batch]
                )
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO memories
                            (id, owner_id, scope, type, content, description, tags,
                             metadata, embedding, model_id, importance,
                             created_at, updated_at, last_accessed_at, archived_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        [
                            (
                                r["id"], r["owner_id"], r["scope"], r["type"],
                                r["content"], r["description"], r["tags"],
                                Jsonb(r["metadata"]), Vector(v), embedder.model_id,
                                r["importance"], r["created_at"], r["updated_at"],
                                r["last_accessed_at"], r["archived_at"],
                            )
                            for r, v in zip(batch, vecs)
                        ],
                    )
                    inserted += cur.rowcount
        return {"total": len(rows), "inserted": inserted, "skipped": len(rows) - inserted}
    finally:
        await pool.close()


async def _batches(conn, batch_size: int) -> AsyncIterator[list[dict[str, Any]]]:
    last = None
    while True:
        where = "WHERE id > %s" if last else ""
        rows = await (
            await conn.execute(
                f"SELECT id, description, content FROM memories {where} "
                "ORDER BY id LIMIT %s",
                (last, batch_size) if last else (batch_size,),
            )
        ).fetchall()
        if not rows:
            return
        yield rows
        last = rows[-1]["id"]


async def reindex(settings: Settings, batch_size: int = 64) -> dict[str, Any]:
    """Re-embed every memory (archived included) with the configured model.

    This is the migration path init_schema's model pin points at: it swaps the
    vector column to the new dimension, rebuilds the HNSW index, and updates
    the hm_meta pin — all in one transaction, so a failure changes nothing.
    Restart the server with the new HM_EMBEDDING_* settings afterwards; until
    then a running server still embeds with the old model and will error.
    """
    embedder = create_embedder(
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dim,
        base_url=settings.ollama_base_url,
    )
    count = 0
    async with await psycopg.AsyncConnection.connect(
        settings.database_url, row_factory=dict_row
    ) as conn:
        exists = await (
            await conn.execute("SELECT to_regclass('memories') AS t")
        ).fetchone()
        if exists["t"] is None:
            raise RuntimeError("no memory store found — start the server once first")
        await register_vector_async(conn)
        async with conn.transaction():
            await conn.execute(
                f"ALTER TABLE memories ADD COLUMN embedding_new vector({embedder.dim})"
            )
            async for rows in _batches(conn, batch_size):
                vecs = embedder.embed_documents(
                    [record_text(r["description"], r["content"]) for r in rows]
                )
                async with conn.cursor() as cur:
                    await cur.executemany(
                        "UPDATE memories SET embedding_new = %s, model_id = %s "
                        "WHERE id = %s",
                        [
                            (Vector(v), embedder.model_id, r["id"])
                            for r, v in zip(rows, vecs)
                        ],
                    )
                count += len(rows)
            # Dropping the old column also drops its HNSW index.
            await conn.execute("ALTER TABLE memories DROP COLUMN embedding")
            await conn.execute(
                "ALTER TABLE memories RENAME COLUMN embedding_new TO embedding"
            )
            await conn.execute(
                "ALTER TABLE memories ALTER COLUMN embedding SET NOT NULL"
            )
            await conn.execute(
                "CREATE INDEX memories_embedding_idx "
                "ON memories USING hnsw (embedding vector_cosine_ops)"
            )
            await conn.execute(
                """
                INSERT INTO hm_meta (key, value)
                VALUES ('embedding_dim', %s), ('model_id', %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (str(embedder.dim), embedder.model_id),
            )
    return {"reindexed": count, "model": embedder.model_id, "dim": embedder.dim}
