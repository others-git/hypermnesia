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
    "importance, created_at, updated_at, last_accessed_at, archived_at"
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
                    WHERE scope = %s AND archived_at IS NULL
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

    def _rank_score(self, relevance: float, importance: float, last_accessed: datetime | None) -> float:
        """Blend a relevance signal with recency decay and importance.

        ``relevance`` is in [0,1]: raw cosine similarity for pure-vector search, or
        the fused reciprocal-rank score for hybrid search. Recency decays
        exponentially from ``last_accessed_at`` with the configured half-life;
        importance is normalised to [0,1] against ``importance_cap``.
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
            s.score_weight_similarity * relevance
            + s.score_weight_recency * recency
            + s.score_weight_importance * imp
        )

    def _fuse(
        self,
        vrows: list[dict[str, Any]],
        lrows: list[dict[str, Any]],
        floor: float,
        k: int,
    ) -> list[SearchHit]:
        """Reciprocal-rank-fuse vector + lexical candidates, then blend and rank.

        ``vrows`` are ordered by vector distance, ``lrows`` by lexical rank; each
        row carries cosine ``similarity``. The similarity floor drops purely
        semantic candidates, but a lexical match bypasses it (an exact-token hit is
        intentional relevance, not noise). Returns the top ``k`` SearchHits.
        """
        s = self.settings
        kk = s.rrf_k
        wv, wl = s.hybrid_vector_weight, s.hybrid_lexical_weight
        vrank = {r["id"]: i for i, r in enumerate(vrows, 1)}
        lrank = {r["id"]: i for i, r in enumerate(lrows, 1)}
        rows_by_id: dict[str, dict[str, Any]] = {}
        for r in (*vrows, *lrows):
            rows_by_id.setdefault(r["id"], r)
        # Normalise fused score to [0,1] against the best possible (rank 1 in every
        # list that returned anything) so it stays comparable to the blend weights.
        max_rrf = (wv / (kk + 1) if vrows else 0.0) + (wl / (kk + 1) if lrows else 0.0)
        if max_rrf == 0.0:
            return []

        hits: list[SearchHit] = []
        for mid, row in rows_by_id.items():
            is_lexical = mid in lrank
            if not is_lexical and row["similarity"] < floor:
                continue
            rrf = 0.0
            if mid in vrank:
                rrf += wv / (kk + vrank[mid])
            if mid in lrank:
                rrf += wl / (kk + lrank[mid])
            hit = SearchHit(**row)
            hit.score = self._rank_score(rrf / max_rrf, hit.importance, hit.last_accessed_at)
            hits.append(hit)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

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
        # Pull a wider candidate pool than k from each signal, then fuse + re-rank
        # in Python so recency/importance can reorder within the neighbourhood.
        pool = max(k * self.settings.rerank_candidate_multiplier, k)
        vec = Vector(self.embedder.embed_query(query))
        # HNSW post-filters on scope/tags during the index walk, so a filtered
        # search can return fewer than `pool` rows. Widen ef_search past the pool
        # (txn-local) so the scan keeps enough candidates to fill it. Capped at the
        # pgvector max of 1000.
        ef_search = min(max(pool * 2, 40), 1000)
        params = {"q": vec, "scopes": scopes, "tags": tags, "pool": pool, "query": query}
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            await conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            vrows = await (
                await conn.execute(
                    f"""
                    SELECT {_SELECT_COLS}, 1 - (embedding <=> %(q)s) AS similarity
                    FROM memories
                    WHERE scope = ANY(%(scopes)s)
                      AND archived_at IS NULL
                      AND (%(tags)s::text[] IS NULL OR tags && %(tags)s)
                    ORDER BY embedding <=> %(q)s
                    LIMIT %(pool)s
                    """,
                    params,
                )
            ).fetchall()

            lrows: list[dict[str, Any]] = []
            if self.settings.hybrid_search:
                # websearch_to_tsquery tolerates arbitrary user text (no syntax
                # errors); an all-stopword query yields no matches -> pure vector.
                lrows = await (
                    await conn.execute(
                        f"""
                        SELECT {_SELECT_COLS}, 1 - (embedding <=> %(q)s) AS similarity
                        FROM memories
                        WHERE scope = ANY(%(scopes)s)
                          AND archived_at IS NULL
                          AND (%(tags)s::text[] IS NULL OR tags && %(tags)s)
                          AND content_tsv @@ websearch_to_tsquery('english', %(query)s)
                        ORDER BY ts_rank_cd(content_tsv,
                                 websearch_to_tsquery('english', %(query)s)) DESC
                        LIMIT %(pool)s
                        """,
                        params,
                    )
                ).fetchall()

            hits = self._fuse(vrows, lrows, floor, k)
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
                    "WHERE id = %s AND scope = ANY(%s) AND archived_at IS NULL",
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
                    "WHERE id = %s AND scope = ANY(%s) AND archived_at IS NULL",
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
                      AND archived_at IS NULL
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

    async def forget(
        self,
        scopes: list[str],
        *,
        tags: list[str] | None = None,
        older_than_days: float,
        importance_floor: float,
        apply: bool = False,
    ) -> dict[str, Any]:
        """Archive (soft-delete) stale, low-importance memories.

        A memory is eligible when it hasn't been recalled in ``older_than_days``
        AND its ``importance`` is at or below ``importance_floor`` — so recall
        (which bumps ``last_accessed_at``) and a higher importance both protect it.
        With ``apply=False`` (default) this only reports candidates; with
        ``apply=True`` it sets ``archived_at`` so they drop out of recall.
        """
        empty = {"dry_run": not apply, "scopes": scopes, "matched": 0, "memories": []}
        if not scopes:
            return empty
        where = (
            "scope = ANY(%(scopes)s) AND archived_at IS NULL "
            "AND last_accessed_at < now() - (%(days)s::text || ' days')::interval "
            "AND importance <= %(imp)s "
            "AND (%(tags)s::text[] IS NULL OR tags && %(tags)s)"
        )
        params = {
            "scopes": scopes,
            "days": older_than_days,
            "imp": importance_floor,
            "tags": tags,
        }
        async with self.pool.connection() as conn:
            conn.row_factory = dict_row
            if apply:
                rows = await (
                    await conn.execute(
                        f"UPDATE memories SET archived_at = now() WHERE {where} "
                        "RETURNING id::text, description, importance, last_accessed_at",
                        params,
                    )
                ).fetchall()
            else:
                rows = await (
                    await conn.execute(
                        f"SELECT id::text, description, importance, last_accessed_at "
                        f"FROM memories WHERE {where} ORDER BY last_accessed_at LIMIT 100",
                        params,
                    )
                ).fetchall()
        return {
            "dry_run": not apply,
            "scopes": scopes,
            "matched": len(rows),
            "memories": [
                {
                    "id": r["id"],
                    "description": r["description"],
                    "importance": r["importance"],
                    "last_accessed_at": r["last_accessed_at"].isoformat()
                    if r["last_accessed_at"]
                    else None,
                }
                for r in rows
            ],
        }
