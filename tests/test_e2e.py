"""End-to-end integration tests against a running hypermnesia MCP server.

Black-box: everything goes through the MCP tools over HTTP, exercising the full
stack (FastMCP -> auth -> service -> pgvector -> embeddings).

Run against the docker-compose stack:

    docker compose up -d --build
    pytest tests/test_e2e.py            # needs fastmcp + pytest-asyncio

Skipped automatically if the server isn't reachable.
"""

from __future__ import annotations

import uuid

import pytest

from conftest import E2E_SCOPE, E2E_URL, make_client

pytest.importorskip("fastmcp")

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
async def _server_up():
    try:
        async with make_client() as c:
            await c.list_tools()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"hypermnesia server not reachable at {E2E_URL}: {e}")


@pytest.fixture
async def client():
    async with make_client() as c:
        yield c


@pytest.fixture
def tag() -> str:
    return "t-" + uuid.uuid4().hex[:10]


async def _save(client, tag, content, description, **kw):
    r = await client.call_tool(
        "memory_save",
        {
            "content": content,
            "description": description,
            "scope": E2E_SCOPE,
            "tags": [tag],
            **kw,
        },
    )
    return r.data


async def test_lists_all_tools(client):
    names = {t.name for t in await client.list_tools()}
    assert {
        "memory_save",
        "memory_search",
        "memory_get",
        "memory_list",
        "memory_delete",
        "memory_update",
        "memory_forget",
    } <= names


async def test_full_crud_lifecycle(client, tag):
    # CREATE
    created = await _save(
        client,
        tag,
        "Project hypermnesia is an MCP-based semantic memory store for agents.",
        "hypermnesia = MCP semantic memory store",
        type="project",
    )
    assert created["created"] is True
    mid = created["memory"]["id"]
    assert created["memory"]["owner_id"]  # stamped from the principal

    # READ (get)
    got = (await client.call_tool("memory_get", {"memory_id": mid})).data
    assert got is not None and got["id"] == mid
    assert tag in got["tags"]
    assert got["type"] == "project"

    # READ (list, filtered by our unique tag)
    listed = (
        await client.call_tool("memory_list", {"scope": E2E_SCOPE, "tags": [tag]})
    ).data
    assert any(m["id"] == mid for m in listed)

    # DELETE
    deleted = (await client.call_tool("memory_delete", {"memory_id": mid})).data
    assert deleted["deleted"] is True

    # gone
    assert (await client.call_tool("memory_get", {"memory_id": mid})).data is None
    # deleting again is a no-op
    assert (await client.call_tool("memory_delete", {"memory_id": mid})).data[
        "deleted"
    ] is False


async def test_semantic_recall_ranks_relevant_first(client, tag):
    ids = []
    try:
        ids.append(
            (
                await _save(
                    client,
                    tag,
                    "The user wants embedding models to run locally on CPU and be pluggable.",
                    "User prefers local CPU pluggable embeddings",
                )
            )["memory"]["id"]
        )
        ids.append(
            (
                await _save(
                    client,
                    tag,
                    "Deployment uses docker-compose with Postgres and pgvector.",
                    "Deploy via docker-compose + pgvector",
                )
            )["memory"]["id"]
        )

        hits = (
            await client.call_tool(
                "memory_search",
                {
                    "query": "what embedding hardware and setup does the user want?",
                    "scope": E2E_SCOPE,
                    "tags": [tag],
                    "k": 2,
                    "min_similarity": 0.0,  # independent of the configured floor
                },
            )
        ).data
        assert len(hits) == 2
        # the embeddings memory should rank above the deployment one
        assert "embedding" in hits[0]["description"].lower()
        assert hits[0]["similarity"] >= hits[1]["similarity"]
        assert 0.0 <= hits[0]["similarity"] <= 1.0
    finally:
        for mid in ids:
            await client.call_tool("memory_delete", {"memory_id": mid})


async def test_search_returns_blended_score(client, tag):
    saved = await _save(
        client,
        tag,
        "The user wants embeddings to run locally on CPU.",
        "local CPU embeddings",
    )
    try:
        hits = (
            await client.call_tool(
                "memory_search",
                {"query": "where do embeddings run?", "scope": E2E_SCOPE, "tags": [tag]},
            )
        ).data
        assert hits
        h = hits[0]
        # both raw similarity and the blended rank score are exposed
        assert 0.0 <= h["similarity"] <= 1.0
        assert h["score"] >= h["similarity"]  # recency+importance only add
        # results are ordered by the blended score, descending
        scores = [x["score"] for x in hits]
        assert scores == sorted(scores, reverse=True)
    finally:
        await client.call_tool("memory_delete", {"memory_id": saved["memory"]["id"]})


async def test_hybrid_surfaces_exact_token_over_floor(client, tag):
    # A distinctive token embeddings handle poorly, in otherwise unrelated text.
    saved = await _save(
        client,
        tag,
        "Internal ticket Zxqv9931Q tracks the storage layer rewrite.",
        "ticket Zxqv9931Q storage rewrite",
    )
    try:
        # An aggressive floor would drop a pure-vector match, but the lexical
        # (keyword) side bypasses the floor for an exact token hit.
        hits = (
            await client.call_tool(
                "memory_search",
                {
                    "query": "Zxqv9931Q",
                    "scope": E2E_SCOPE,
                    "tags": [tag],
                    "min_similarity": 0.9,
                },
            )
        ).data
        assert any(h["id"] == saved["memory"]["id"] for h in hits)
    finally:
        await client.call_tool("memory_delete", {"memory_id": saved["memory"]["id"]})


async def test_min_similarity_filters_weak_matches(client, tag):
    saved = await _save(
        client,
        tag,
        "Postgres connection pooling settings for the service.",
        "pg connection pooling",
    )
    try:
        # An unrelated query with an aggressive floor should return nothing.
        hits = (
            await client.call_tool(
                "memory_search",
                {
                    "query": "favourite pizza toppings",
                    "scope": E2E_SCOPE,
                    "tags": [tag],
                    "min_similarity": 0.95,
                },
            )
        ).data
        assert hits == []
    finally:
        await client.call_tool("memory_delete", {"memory_id": saved["memory"]["id"]})


async def test_update_changes_only_given_fields(client, tag):
    saved = await _save(
        client,
        tag,
        "Original content about deployment.",
        "deployment note",
        type="fact",
        importance=1.0,
    )
    mid = saved["memory"]["id"]
    try:
        updated = (
            await client.call_tool(
                "memory_update",
                {"memory_id": mid, "importance": 3.0, "type": "project"},
            )
        ).data
        assert updated["id"] == mid
        assert updated["importance"] == 3.0
        assert updated["type"] == "project"
        # untouched fields are preserved
        assert updated["content"] == "Original content about deployment."
        assert tag in updated["tags"]

        # content edit re-embeds and is retrievable by its new meaning
        edited = (
            await client.call_tool(
                "memory_update",
                {"memory_id": mid, "content": "Now about caching strategy instead."},
            )
        ).data
        assert edited["content"] == "Now about caching strategy instead."
    finally:
        await client.call_tool("memory_delete", {"memory_id": mid})


async def test_update_unknown_id_returns_null(client):
    res = (
        await client.call_tool(
            "memory_update", {"memory_id": str(uuid.uuid4()), "importance": 2.0}
        )
    ).data
    assert res is None


async def test_forget_archives_stale_low_importance(client, tag):
    # low importance -> eligible; high importance -> protected.
    a = await _save(
        client, tag, "ephemeral scratch note about a temporary thing",
        "scratch note", importance=0.5,
    )
    b = await _save(
        client, tag, "a pinned decision we must keep around long term",
        "pinned decision", importance=5.0,
    )
    aid, bid = a["memory"]["id"], b["memory"]["id"]
    cid = None
    try:
        # older_than_days=0 ignores the age gate, isolating the importance gate.
        forget_args = {
            "scope": E2E_SCOPE, "tags": [tag],
            "older_than_days": 0, "importance_floor": 1.0,
        }
        # dry run: a is a candidate, b is protected, and nothing is archived yet.
        dry = (await client.call_tool("memory_forget", forget_args)).data
        assert dry["dry_run"] is True
        cand = {m["id"] for m in dry["memories"]}
        assert aid in cand and bid not in cand
        assert (await client.call_tool("memory_get", {"memory_id": aid})).data is not None

        # apply: a gets archived.
        applied = (
            await client.call_tool("memory_forget", {**forget_args, "apply": True})
        ).data
        assert applied["dry_run"] is False
        assert aid in {m["id"] for m in applied["memories"]}

        # a is now invisible to get/list (and recall); b remains.
        assert (await client.call_tool("memory_get", {"memory_id": aid})).data is None
        assert (await client.call_tool("memory_get", {"memory_id": bid})).data is not None
        listed = {
            m["id"]
            for m in (
                await client.call_tool("memory_list", {"scope": E2E_SCOPE, "tags": [tag]})
            ).data
        }
        assert aid not in listed and bid in listed

        # saving a near-duplicate of the archived note inserts fresh (no revival).
        c = await _save(
            client, tag, "ephemeral scratch note about a temporary thing",
            "scratch note", importance=0.5,
        )
        assert c["created"] is True and c["memory"]["id"] != aid
        cid = c["memory"]["id"]
    finally:
        for mid in (aid, bid, cid):
            if mid:
                await client.call_tool("memory_delete", {"memory_id": mid})


async def test_save_dedupes_near_duplicate(client, tag):
    a = await _save(
        client,
        tag,
        "The user prefers pluggable local CPU embedding models.",
        "pluggable local CPU embeddings",
    )
    assert a["created"] is True
    try:
        b = await _save(
            client,
            tag,
            "The user prefers pluggable local CPU embedding models too.",
            "pluggable local CPU embeddings",
        )
        # near-duplicate within the same scope updates instead of inserting
        assert b["created"] is False
        assert b["memory"]["id"] == a["memory"]["id"]
        # the pre-merge memory is surfaced so a bad overwrite is catchable
        assert b["replaced"]["id"] == a["memory"]["id"]
        assert b["replaced"]["content"] == a["memory"]["content"]
        # a fresh insert reports no replacement
        assert "replaced" not in a
    finally:
        await client.call_tool("memory_delete", {"memory_id": a["memory"]["id"]})


async def test_write_to_unauthorized_scope_is_rejected(client):
    with pytest.raises(Exception) as exc:
        await client.call_tool(
            "memory_save",
            {"content": "x", "description": "y", "scope": "user:not-allowed"},
        )
    assert "scope" in str(exc.value).lower()


async def test_unauthenticated_call_is_rejected():
    async with make_client(token=None) as c:
        with pytest.raises(Exception) as exc:
            await c.call_tool("memory_list", {"scope": E2E_SCOPE})
    msg = str(exc.value).lower()
    assert "authorization" in msg or "bearer" in msg


async def test_projects_do_not_trample_via_roots(tag):
    """Two sessions with different workspace roots are isolated automatically,
    with no explicit scope passed — the trampling scenario for a global config."""
    alpha = [f"file:///workspace/alpha-{tag}"]
    beta = [f"file:///workspace/beta-{tag}"]

    # Project alpha saves a memory with NO explicit scope.
    async with make_client(roots=alpha) as a:
        saved = await a.call_tool(
            "memory_save",
            {"content": "alpha-only secret value", "description": f"alpha {tag}",
             "tags": [tag]},
        )
        aid = saved.data["memory"]["id"]
        assert saved.data["memory"]["scope"].startswith("project:alpha-")
        # alpha recalls its own memory (no scope passed)
        hits = (
            await a.call_tool("memory_search", {"query": "secret value", "tags": [tag]})
        ).data
        assert any(h["id"] == aid for h in hits)

    try:
        # Project beta must NOT see alpha's memory.
        async with make_client(roots=beta) as b:
            hits = (
                await b.call_tool("memory_search", {"query": "secret value", "tags": [tag]})
            ).data
            assert all(h["id"] != aid for h in hits)
            # beta also can't fetch it by id (different project)
            assert (await b.call_tool("memory_get", {"memory_id": aid})).data is None
    finally:
        async with make_client(roots=alpha) as a:
            await a.call_tool("memory_delete", {"memory_id": aid})


async def test_shared_scope_is_visible_across_projects(tag):
    """Explicit `scope: shared` crosses project boundaries on purpose."""
    alpha = [f"file:///workspace/alpha-{tag}"]
    beta = [f"file:///workspace/beta-{tag}"]
    async with make_client(roots=alpha) as a:
        saved = await a.call_tool(
            "memory_save",
            {"content": "company-wide convention", "description": f"shared {tag}",
             "scope": "shared", "tags": [tag]},
        )
        sid = saved.data["memory"]["id"]
        assert saved.data["memory"]["scope"] == "shared"
    try:
        async with make_client(roots=beta) as b:
            hits = (
                await b.call_tool("memory_search", {"query": "convention", "tags": [tag]})
            ).data
            assert any(h["id"] == sid for h in hits)
    finally:
        async with make_client(roots=alpha) as a:
            await a.call_tool("memory_delete", {"memory_id": sid})


async def test_project_header_override(tag):
    """The X-Hypermnesia-Project header gives a stable key two clients can share."""
    key = f"team-{tag}"
    async with make_client(project_header=key) as a:
        saved = await a.call_tool(
            "memory_save",
            {"content": "team memory", "description": f"team {tag}", "tags": [tag]},
        )
        mid = saved.data["memory"]["id"]
        assert saved.data["memory"]["scope"] == f"project:team-{tag}"
    try:
        # a different client using the same header key sees it
        async with make_client(project_header=key) as b:
            assert (await b.call_tool("memory_get", {"memory_id": mid})).data is not None
    finally:
        async with make_client(project_header=key) as a:
            await a.call_tool("memory_delete", {"memory_id": mid})


async def test_search_only_returns_accessible_scopes(client, tag):
    # Save in an allowed scope, then confirm a disallowed scope filter is refused.
    saved = await _save(client, tag, "scoped memory content", "scoped memory")
    try:
        with pytest.raises(Exception) as exc:
            await client.call_tool(
                "memory_search",
                {"query": "anything", "scope": "user:not-allowed", "k": 5},
            )
        assert "scope" in str(exc.value).lower()
    finally:
        await client.call_tool("memory_delete", {"memory_id": saved["memory"]["id"]})
