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
        "memory_restore",
        "memory_stats",
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
                    # Both gates off: this asserts ranking order, not filtering.
                    "min_similarity": 0.0,
                    "relative_cutoff": 0.0,
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


async def test_relative_cutoff_trims_the_near_misses(client, tag):
    """A clear winner should not arrive buried in loosely-related memories.

    The absolute floor cannot do this: unrelated memories score well inside
    bge-small's cosine range, so a floor loose enough to keep the answer keeps
    them too. Distance to the best hit separates them.
    """
    winner = await _save(
        client,
        tag,
        "The deploy pipeline pushes to GHCR only on tags matching v*, never on "
        "branch pushes; a branch push builds the image but skips the registry.",
        "deploy pushes to GHCR on version tags only",
    )
    others = []
    for content, desc in [
        ("Postgres connection pooling uses psycopg_pool with max_size 10.",
         "pg connection pooling settings"),
        ("Run the unit suite with pytest -m 'not e2e'; it needs no infrastructure.",
         "running the unit test suite"),
        ("The embedding model is pinned in hm_meta on first run and cannot change.",
         "embedding model is pinned in the store"),
    ]:
        others.append(await _save(client, tag, content, desc))
    ids = [winner["memory"]["id"]] + [o["memory"]["id"] for o in others]
    try:
        args = {"query": "when does CI publish the image to the registry",
                "scope": E2E_SCOPE, "tags": [tag], "min_similarity": 0.0}
        wide = (await client.call_tool(
            "memory_search", {**args, "relative_cutoff": 0.0})).data
        tight = (await client.call_tool("memory_search", args)).data  # configured default

        assert len(wide) == 4, "floor alone keeps every near-miss"
        assert tight[0]["id"] == winner["memory"]["id"]
        assert len(tight) < len(wide), "the relative gate trims the near-misses"
    finally:
        for mid in ids:
            await client.call_tool("memory_delete", {"memory_id": mid})


async def test_relative_cutoff_of_zero_is_an_escape_hatch(client, tag):
    # An agent wanting a broad survey rather than the best answer can turn the
    # gate off per search, and then sees everything the floor admits.
    saved = [
        await _save(client, tag, "Rate limiting is enforced per principal.",
                    "per-principal rate limits"),
        await _save(client, tag, "Bread dough proofs for 12 hours at 4 degrees.",
                    "sourdough cold proofing schedule"),
    ]
    ids = [x["memory"]["id"] for x in saved]
    try:
        hits = (
            await client.call_tool(
                "memory_search",
                {"query": "rate limits", "scope": E2E_SCOPE, "tags": [tag],
                 "min_similarity": 0.0, "relative_cutoff": 0.0},
            )
        ).data
        assert {h["id"] for h in hits} == set(ids)
    finally:
        for mid in ids:
            await client.call_tool("memory_delete", {"memory_id": mid})


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


async def test_forget_matched_reports_true_total_beyond_limit(client, tag):
    """A dry run must never under-report what an apply would archive: `matched`
    is the true total even when `memories` is cut to `limit`."""
    # Distinct topics, or the server's own dedup would merge them on save.
    notes = [
        ("staging runs kubernetes 1.29 on three nodes", "staging k8s version"),
        ("the marketing site is built with a static generator", "marketing site stack"),
        ("friday lunch orders go to the taco place", "friday lunch spot"),
    ]
    ids = []
    try:
        for content, description in notes:
            saved = await _save(client, tag, content, description, importance=0.5)
            assert saved["created"] is True
            ids.append(saved["memory"]["id"])

        forget_args = {
            "scope": E2E_SCOPE, "tags": [tag],
            "older_than_days": 0, "importance_floor": 1.0, "limit": 2,
        }
        dry = (await client.call_tool("memory_forget", forget_args)).data
        assert dry["dry_run"] is True
        assert dry["matched"] == 3
        assert len(dry["memories"]) == 2
        assert dry["truncated"] is True

        # apply archives everything matched, not just the listed sample
        applied = (
            await client.call_tool("memory_forget", {**forget_args, "apply": True})
        ).data
        assert applied["matched"] == 3
        assert len(applied["memories"]) == 2
        assert applied["truncated"] is True
        for mid in ids:
            assert (await client.call_tool("memory_get", {"memory_id": mid})).data is None

        # nothing left: an un-truncated empty report
        again = (await client.call_tool("memory_forget", forget_args)).data
        assert again["matched"] == 0 and again["truncated"] is False
    finally:
        for mid in ids:
            await client.call_tool("memory_delete", {"memory_id": mid})


async def test_restore_unarchives_memory(client, tag):
    saved = await _save(
        client, tag, "a note that gets forgotten then brought back",
        "restorable note", importance=0.5,
    )
    mid = saved["memory"]["id"]
    try:
        # forget it
        await client.call_tool(
            "memory_forget",
            {"scope": E2E_SCOPE, "tags": [tag], "older_than_days": 0,
             "importance_floor": 1.0, "apply": True},
        )
        assert (await client.call_tool("memory_get", {"memory_id": mid})).data is None

        # it's hidden from a normal list but visible with include_archived
        plain = {
            m["id"]
            for m in (
                await client.call_tool("memory_list", {"scope": E2E_SCOPE, "tags": [tag]})
            ).data
        }
        assert mid not in plain
        archived = (
            await client.call_tool(
                "memory_list",
                {"scope": E2E_SCOPE, "tags": [tag], "include_archived": True},
            )
        ).data
        entry = next(m for m in archived if m["id"] == mid)
        assert entry["archived_at"] is not None

        # restore brings it back and clears archived_at
        restored = (await client.call_tool("memory_restore", {"memory_id": mid})).data
        assert restored["id"] == mid and restored["archived_at"] is None
        assert (await client.call_tool("memory_get", {"memory_id": mid})).data is not None

        # restore refreshed last_accessed_at, so a realistic sweep won't re-archive it
        dry = (
            await client.call_tool(
                "memory_forget",
                {"scope": E2E_SCOPE, "tags": [tag], "older_than_days": 1,
                 "importance_floor": 1.0},
            )
        ).data
        assert mid not in {m["id"] for m in dry["memories"]}
    finally:
        await client.call_tool("memory_delete", {"memory_id": mid})


async def test_list_is_an_index_and_omits_content(client, tag):
    """`memory_list` is for orientation, so it must not carry every body.

    Content runs ~11x longer than the description; including it made the default
    limit=50 call cost roughly four times what the index itself costs.
    """
    saved = await _save(
        client,
        tag,
        "A deliberately long body that a caller scanning an index has no use for. " * 6,
        "short one-line description",
    )
    mid = saved["memory"]["id"]
    try:
        (row,) = (
            await client.call_tool("memory_list", {"scope": E2E_SCOPE, "tags": [tag]})
        ).data
        assert "content" not in row
        assert row["description"] == "short one-line description"
        # the index still carries everything needed to pick and then fetch
        assert row["id"] == mid and "importance" in row and "tags" in row
        # ...and memory_get is the way to get the body
        assert "deliberately long body" in (
            await client.call_tool("memory_get", {"memory_id": mid})
        ).data["content"]
    finally:
        await client.call_tool("memory_delete", {"memory_id": mid})


async def test_list_full_true_restores_content(client, tag):
    saved = await _save(client, tag, "the whole body", "desc for full listing")
    mid = saved["memory"]["id"]
    try:
        (row,) = (
            await client.call_tool(
                "memory_list", {"scope": E2E_SCOPE, "tags": [tag], "full": True}
            )
        ).data
        assert row["content"] == "the whole body"
    finally:
        await client.call_tool("memory_delete", {"memory_id": mid})


async def test_restore_unknown_id_returns_null(client):
    res = (
        await client.call_tool("memory_restore", {"memory_id": str(uuid.uuid4())})
    ).data
    assert res is None


async def test_search_logging_feeds_stats(client, tag):
    """Every search is logged; memory_stats turns the log into recall metrics.

    The server is shared, so all assertions are deltas / containment, never
    exact totals.
    """
    before = (await client.call_tool("memory_stats", {"days": 1})).data

    saved = await _save(
        client, tag,
        "The deploy pipeline uses a blue-green cutover behind the load balancer.",
        "blue-green deploy cutover",
    )
    miss_query = f"zzmiss{tag}"  # matches nothing lexically or semantically
    try:
        hit_search = (
            await client.call_tool(
                "memory_search",
                {"query": "how do deploys cut over?", "scope": E2E_SCOPE,
                 "tags": [tag], "min_similarity": 0.0},
            )
        ).data
        assert hit_search  # this one must land so the stats see a non-empty search

        empty_search = (
            await client.call_tool(
                "memory_search",
                {"query": miss_query, "scope": E2E_SCOPE, "min_similarity": 0.99},
            )
        ).data
        assert empty_search == []

        after = (await client.call_tool("memory_stats", {"days": 1})).data
        s = after["searches"]
        assert s["total"] >= before["searches"]["total"] + 2
        assert s["empty"] >= before["searches"]["empty"] + 1
        assert 0.0 <= s["empty_rate"] <= 1.0
        assert s["avg_hits"] is not None and s["avg_top_score"] is not None
        assert s["p50_latency_ms"] > 0 and s["p95_latency_ms"] >= s["p50_latency_ms"]

        # the miss shows up as an investigatable recall failure
        assert miss_query in {q["query"] for q in after["recent_empty_queries"]}

        # store-size side: our memory is counted in its scope
        assert after["memories"]["active"] >= 1
        assert any(r["scope"] == E2E_SCOPE for r in after["memories"]["by_scope"])
    finally:
        await client.call_tool("memory_delete", {"memory_id": saved["memory"]["id"]})


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


async def test_same_description_different_fact_is_not_clobbered(client, tag):
    """Two different facts under the same description must coexist — the
    content dedup gate keeps the first from being silently overwritten."""
    a = await _save(
        client, tag,
        "Production deploys go to the Frankfurt region on Fridays.",
        "deploy target",
    )
    b = await _save(
        client, tag,
        "The staging cluster is a minikube VM on the office NAS.",
        "deploy target",
    )
    ids = {a["memory"]["id"], b["memory"]["id"]}
    try:
        assert b["created"] is True
        assert len(ids) == 2
        listed = (
            await client.call_tool("memory_list", {"scope": E2E_SCOPE, "tags": [tag]})
        ).data
        assert ids <= {m["id"] for m in listed}
    finally:
        for mid in ids:
            await client.call_tool("memory_delete", {"memory_id": mid})


async def test_usage_guide_is_generated_from_the_live_server(client):
    """The guide must describe *this* server, not a copy of a README."""
    g = (await client.call_tool("memory_usage_guide", {})).data
    assert set(g) == {"claude_md", "setup", "session", "warnings"}

    # Generated from the live tool registry: every registered tool is named.
    names = {t.name for t in await client.list_tools()}
    for name in names:
        assert name in g["claude_md"], f"{name} missing from generated guidance"

    # ...and from live settings, not a hardcoded string.
    assert "hypermnesia" in g["claude_md"]
    assert "/mcp" in g["setup"]
    assert g["session"]["principal"]
    assert g["session"]["scopes_you_can_read"]


async def test_usage_guide_sections_and_bad_input(client):
    only = (await client.call_tool("memory_usage_guide", {"section": "setup"})).data
    assert set(only) == {"setup", "warnings"}
    with pytest.raises(Exception) as exc:
        await client.call_tool("memory_usage_guide", {"section": "nope"})
    assert "unknown section" in str(exc.value)


async def test_usage_guide_reports_the_sessions_own_project_scope(tag):
    key = f"guide-{tag}"
    async with make_client(project_header=key) as c:
        g = (await c.call_tool("memory_usage_guide", {"section": "session"})).data
        assert g["session"]["project_scope"] == f"project:{key}"
        assert g["session"]["project_isolation"] == "active"
        assert f"project:{key}" not in str(g["warnings"])


async def test_usage_guide_warns_when_project_isolation_is_off(client):
    """The default-scope fallback is the silent failure worth shouting about."""
    g = (await client.call_tool("memory_usage_guide", {"section": "session"})).data
    if g["session"]["project_scope"] != "default":
        pytest.skip("this session resolved a real project scope")
    assert g["session"]["project_isolation"].startswith("NOT ACTIVE")
    assert any("NOT isolated" in w for w in g["warnings"])


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


async def test_projects_do_not_trample_via_header(tag):
    """Two sessions keyed by different X-Hypermnesia-Project headers are isolated
    with no explicit scope passed — the trampling scenario for a global config.

    The header is the mechanism that works on every transport, so this is the
    guaranteed-coverage twin of the roots test below.
    """
    alpha, beta = f"alpha-{tag}", f"beta-{tag}"

    async with make_client(project_header=alpha) as a:
        saved = await a.call_tool(
            "memory_save",
            {"content": "alpha-only secret value", "description": f"alpha {tag}",
             "tags": [tag]},
        )
        aid = saved.data["memory"]["id"]
        assert saved.data["memory"]["scope"] == f"project:{alpha}"
        hits = (
            await a.call_tool("memory_search", {"query": "secret value", "tags": [tag]})
        ).data
        assert any(h["id"] == aid for h in hits)

    try:
        async with make_client(project_header=beta) as b:
            hits = (
                await b.call_tool("memory_search", {"query": "secret value", "tags": [tag]})
            ).data
            assert all(h["id"] != aid for h in hits)
            assert (await b.call_tool("memory_get", {"memory_id": aid})).data is None
    finally:
        async with make_client(project_header=alpha) as a:
            await a.call_tool("memory_delete", {"memory_id": aid})


async def test_projects_do_not_trample_via_roots(tag):
    """Same isolation, derived automatically from MCP workspace roots.

    Roots are best-effort: the capability is deprecated (SEP-2577) and the
    server cannot request them on a transport with no back-channel for
    server-initiated requests. Where they don't resolve this skips rather than
    fails — the isolation property itself is covered by the header test above.
    """
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
        if saved.data["memory"]["scope"] == "default":
            await a.call_tool("memory_delete", {"memory_id": aid})
            pytest.skip(
                "client/transport did not deliver workspace roots (deprecated "
                "capability, needs a back-channel); header test covers isolation"
            )
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
