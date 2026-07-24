"""End-to-end tests for the admin CLI (export / import / reindex).

These need the `hypermnesia` console script plus DIRECT database access, so
they only run where `HM_TEST_DATABASE_URL` is set — in practice inside the
compose test container, which reaches the stack's db service. Skipped
elsewhere (e.g. running the e2e suite against a remote server over MCP only).
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid

import pytest

from conftest import E2E_SCOPE, E2E_URL, make_client

pytest.importorskip("fastmcp")

DB_URL = os.environ.get("HM_TEST_DATABASE_URL")
ALT_MODEL = os.environ.get("HM_TEST_ALT_MODEL")
DEFAULT_MODEL = os.environ.get("HM_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not DB_URL, reason="HM_TEST_DATABASE_URL not set"),
]


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


def _cli(*args: str, model: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "HM_DATABASE_URL": DB_URL}
    if model:
        env["HM_EMBEDDING_MODEL"] = model
    return subprocess.run(
        ["hypermnesia", *args], capture_output=True, text=True, env=env, timeout=600
    )


async def _save(client, tag, content, description, **kw):
    r = await client.call_tool(
        "memory_save",
        {"content": content, "description": description, "scope": E2E_SCOPE,
         "tags": [tag], **kw},
    )
    return r.data


async def test_export_import_restores_deleted_memory(client, tag, tmp_path):
    a = await _save(
        client, tag,
        "The invoice service retries failed webhooks three times with backoff.",
        "invoice webhook retries",
    )
    b = await _save(
        client, tag,
        "Grafana dashboards live in the ops repo under dashboards/.",
        "grafana dashboard location",
    )
    aid, bid = a["memory"]["id"], b["memory"]["id"]
    dump = tmp_path / "dump.jsonl"
    try:
        r = _cli("export", "--out", str(dump))
        assert r.returncode == 0, r.stderr
        rows = {
            row["id"]: row
            for row in map(json.loads, dump.read_text().splitlines())
        }
        assert aid in rows and bid in rows
        assert rows[aid]["content"] == a["memory"]["content"]
        assert "embedding" not in rows[aid]

        # lose one memory, then restore the dump
        await client.call_tool("memory_delete", {"memory_id": aid})
        assert (await client.call_tool("memory_get", {"memory_id": aid})).data is None

        r = _cli("import", str(dump))
        assert r.returncode == 0, r.stderr
        result = json.loads(r.stdout)
        assert result["inserted"] == 1  # only the deleted row; the rest skipped
        assert result["skipped"] == result["total"] - 1

        # back under its original id, and re-embedded (semantically findable)
        got = (await client.call_tool("memory_get", {"memory_id": aid})).data
        assert got is not None and got["content"] == a["memory"]["content"]
        hits = (
            await client.call_tool(
                "memory_search",
                {"query": "how often do webhooks retry?", "scope": E2E_SCOPE,
                 "tags": [tag], "min_similarity": 0.0},
            )
        ).data
        assert any(h["id"] == aid for h in hits)

        # importing again is a no-op, never an overwrite
        r = _cli("import", str(dump))
        assert json.loads(r.stdout)["inserted"] == 0
    finally:
        for mid in (aid, bid):
            await client.call_tool("memory_delete", {"memory_id": mid})


@pytest.mark.skipif(not ALT_MODEL, reason="HM_TEST_ALT_MODEL not set")
async def test_reindex_swaps_model_and_back(client, tag):
    import psycopg

    def meta() -> dict[str, str]:
        with psycopg.connect(DB_URL) as conn:
            return dict(conn.execute("SELECT key, value FROM hm_meta").fetchall())

    saved = await _save(
        client, tag,
        "Quarterly reports follow the fiscal calendar, not the civil one.",
        "fiscal quarterly reports",
    )
    mid = saved["memory"]["id"]
    try:
        r = _cli("reindex", model=ALT_MODEL)
        assert r.returncode == 0, r.stderr
        out = json.loads(r.stdout)
        assert out["model"] == ALT_MODEL and out["reindexed"] >= 1
        assert meta()["model_id"] == ALT_MODEL
        assert meta()["embedding_dim"] == str(out["dim"])

        # swap back so the store matches the still-running server again
        r = _cli("reindex", model=DEFAULT_MODEL)
        assert r.returncode == 0, r.stderr
        assert json.loads(r.stdout)["model"] == DEFAULT_MODEL
        assert meta()["model_id"] == DEFAULT_MODEL

        # the running server searches the twice-reindexed store fine
        hits = (
            await client.call_tool(
                "memory_search",
                {"query": "which calendar do quarterly reports use?",
                 "scope": E2E_SCOPE, "tags": [tag], "min_similarity": 0.0},
            )
        ).data
        assert any(h["id"] == mid for h in hits)
    finally:
        await client.call_tool("memory_delete", {"memory_id": mid})


async def test_reindex_refuses_missing_store(tmp_path):
    # Point at a real Postgres but a database that has no memories table.
    import psycopg

    dbname = "hm_empty_" + uuid.uuid4().hex[:8]
    admin_url = DB_URL
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{dbname}"')
    try:
        empty_url = admin_url.rsplit("/", 1)[0] + f"/{dbname}"
        env = {**os.environ, "HM_DATABASE_URL": empty_url}
        r = subprocess.run(
            ["hypermnesia", "reindex"], capture_output=True, text=True, env=env,
            timeout=600,
        )
        assert r.returncode != 0
        assert "no memory store found" in r.stderr
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE "{dbname}"')
