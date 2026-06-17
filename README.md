# hypermnesia

A semantic **memory store for AI agents**, exposed over [MCP](https://modelcontextprotocol.io).
Agents call tools to *save* and *recall* persistent memories across sessions instead of
relying on per-session context or hand-edited `CLAUDE.md` files.

- **Recall is semantic**, not key-based — agents search by meaning ("what do I know
  relevant to this task?"), not by knowing an exact key.
- **Local-only, CPU-friendly embeddings.** Default is `fastembed` (ONNX, no PyTorch)
  with `BAAI/bge-small-en-v1.5`. The embedder is pluggable via config.
- **Shared & multi-tenant.** Memories live in scopes; bearer tokens map to a principal
  and the scopes it may read/write. Every query is scope-filtered, so tenants are isolated.
- **Postgres + pgvector** for storage, vectors, and metadata in one place.

## MCP tools

| Tool | Purpose |
|---|---|
| `memory_search(query, scope?, tags?, k)` | Semantic recall (the workhorse) |
| `memory_save(content, description, scope, type?, tags?, metadata?, importance?)` | Store; updates a near-duplicate instead of inserting |
| `memory_get(memory_id)` | Fetch one by id |
| `memory_list(scope?, tags?, limit)` | Browse recent memories (cheap index) |
| `memory_delete(memory_id)` | Delete by id |

`description` is a one-line summary used for ranking and de-duplication — treat it like
the one-liners in Claude Code's `MEMORY.md` index.

## Quick start (Docker)

```bash
cp .env.example .env          # edit HM_AUTH_TOKENS!
docker compose up --build
```

The MCP server listens on `http://localhost:8000/mcp` (streamable HTTP). Point an MCP
client at it with header `Authorization: Bearer <your-token>`.

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Postgres with pgvector (or: docker compose up db)
hypermnesia                   # starts the MCP server
pytest -m "not e2e"           # unit tests only (no DB/network needed)
```

## Tests

- **Unit** (`tests/test_auth.py`, `tests/test_embeddings.py`): pure logic, no infra —
  `pytest -m "not e2e"`.
- **End-to-end** (`tests/test_e2e.py`): black-box CRUD + semantic recall + auth/scope
  isolation, driven through the MCP tools against a running server. They auto-skip if no
  server is reachable.

Run the whole suite against the Docker stack with one command:

```bash
docker compose up -d --build
docker compose --profile test run --rm tests   # builds, waits for health, runs unit + e2e
```

Point the e2e tests elsewhere with `HM_TEST_URL`, `HM_TEST_TOKEN`, `HM_TEST_SCOPE`.

## Configuration

All settings are env vars with the `HM_` prefix (see `.env.example`). Key ones:

| Var | Default | Notes |
|---|---|---|
| `HM_EMBEDDING_PROVIDER` | `fastembed` | `fastembed` \| `sentence_transformers` \| `ollama` |
| `HM_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | any model the provider supports |
| `HM_DEDUPE_THRESHOLD` | `0.92` | cosine sim above which `save` updates vs. inserts |
| `HM_AUTH_TOKENS` | `{}` | `{"token":{"principal":"id","scopes":["..."]}}` |
| `HM_REQUIRE_AUTH` | `true` | when false, all callers are `anonymous`/`default` |

### Swapping the embedding model

Set `HM_EMBEDDING_PROVIDER` / `HM_EMBEDDING_MODEL`. The vector dimension is auto-detected
and **pinned** in the store on first run. Switching to a model with a different dimension
(or a different model entirely) is refused with a clear error, because existing vectors
would no longer be comparable — re-index (dump, drop, reload) when changing models.

To add a new provider, implement the `Embedder` protocol and `@register("name")` it in
`src/hypermnesia/embeddings/providers.py`.

## Status

**v1.** Implemented: MCP tools, pgvector storage, scope-based auth/isolation, semantic
recall, search-before-write de-duplication, and a unit + e2e test suite.

Roadmap: decay/forgetting jobs, hybrid keyword+vector rerank, Redis hot-cache,
per-principal rate limits, web UI.
