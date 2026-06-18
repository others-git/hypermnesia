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
| `memory_search(query, scope?, tags?, k, min_similarity?)` | Hybrid (semantic + keyword) recall; ranks by relevance + recency + importance |
| `memory_save(content, description, scope, type?, tags?, metadata?, importance?)` | Store; updates a near-duplicate instead of inserting |
| `memory_update(memory_id, content?, description?, type?, tags?, metadata?, importance?)` | Edit a known memory in place (only given fields change) |
| `memory_get(memory_id)` | Fetch one by id |
| `memory_list(scope?, tags?, limit, include_archived?)` | Browse recent memories (cheap index); `include_archived` to review forgotten ones |
| `memory_delete(memory_id)` | Delete by id (hard) |
| `memory_forget(scope?, tags?, older_than_days?, importance_floor?, apply?)` | Archive stale, low-importance memories; dry-run unless `apply: true` |
| `memory_restore(memory_id)` | Un-archive a forgotten memory (inverse of `memory_forget`) |

Search results carry both a raw `similarity` (0-1 cosine) and a blended `score` that
adds recency decay (half-life `HM_RECENCY_HALF_LIFE_DAYS`) and normalised `importance`;
tune the mix via `HM_SCORE_WEIGHT_*`. Hits below `HM_SEARCH_MIN_SIMILARITY`
(default `0.4`) are dropped before they reach the agent's context; pass
`min_similarity` to override per search, or `0.0` to disable. The default is tuned
for the bge-small-en-v1.5 cosine range (unrelated text scores ~0.30-0.45, relevant
~0.55+) — re-tune it if you switch embedding models, since the scale changes.

Search is **hybrid**: a vector (semantic) query and a Postgres full-text (keyword)
query are fused with reciprocal-rank fusion, so exact tokens the embedding can't
capture — error codes, flag names, file paths, names — still surface. A pure keyword
hit bypasses the similarity floor on purpose. Toggle with `HM_HYBRID_SEARCH`; tune the
fusion via `HM_RRF_K`, `HM_HYBRID_VECTOR_WEIGHT`, `HM_HYBRID_LEXICAL_WEIGHT`.

**Forgetting.** Stores grow forever and old clutter dilutes recall, so `memory_forget`
archives memories that are both stale (not recalled in `HM_FORGET_AFTER_DAYS`, default
180) and unimportant (`importance <=` `HM_FORGET_IMPORTANCE_FLOOR`, default 1.0).
Recall bumps `last_accessed_at` and a higher `importance` both keep a memory alive, so
anything you use or pin survives. It's a **soft delete** — archived rows drop out of
search/get/list but are kept, not destroyed — and a **dry run by default** (pass
`apply: true` to act). `memory_delete` remains the hard, irreversible removal.
Review what's been archived with `memory_list(include_archived=true)` and bring one
back with `memory_restore(memory_id)` — restoring also refreshes its last-access time
so the next sweep won't immediately re-forget it.

`description` is a one-line summary used for ranking and de-duplication — treat it like
the one-liners in Claude Code's `MEMORY.md` index.

## Quick start (Docker)

The stack is a shared base (`docker-compose.yml`) plus one of two overlays:

```bash
# Release — pull the published image from GHCR (defaults to the `latest` tag):
docker compose -f docker-compose.yml -f docker-compose.release.yml up -d
# pin a version with HM_TAG, e.g. HM_TAG=v0.1.0 docker compose ... up -d

# Dev — build the image from local source:
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

The MCP server listens on `http://localhost:8765/mcp` (streamable HTTP). Point an MCP
client at it with header `Authorization: Bearer <your-token>`.

> Tip: `export COMPOSE_FILE=docker-compose.yml:docker-compose.dev.yml` to drop the
> repeated `-f` flags during development.

## Use it with Claude

Connecting the server is two steps: register it, then tell Claude when to call it.

### Claude Code

Register the running server (HTTP transport, with the bearer token):

```bash
claude mcp add --transport http hypermnesia http://localhost:8765/mcp \
  --header "Authorization: Bearer dev-token" \
  --scope user        # available in every project; use --scope local/project to narrow
```

Verify with `claude mcp list` (should show `connected`); inside a session, `/mcp` lists
the tools. The stack must be running and reachable on the same machine.

Exposing tools isn't enough — Claude won't reach for them unless told when to. Add this
to a `CLAUDE.md` (project-level, or `~/.claude/CLAUDE.md` for all projects):

```markdown
## Persistent memory (hypermnesia MCP)
- At the start of a task, call `memory_search` for relevant prior context.
- When you learn a durable fact, preference, or decision, call `memory_save` with a
  one-line `description` — no `scope` needed; it defaults to this project.
- Pass `scope: "shared"` only for things useful across every project.
- Search before saving; prefer updating a near-duplicate over creating a new memory.
```

You can put this in a **single global** `~/.claude/CLAUDE.md` — memories are partitioned
per project automatically (see below), so projects never trample each other.

### Claude Desktop

`claude_desktop_config.json` is stdio-oriented, so bridge to the HTTP server with
[`mcp-remote`](https://github.com/geelen/mcp-remote):

```json
{
  "mcpServers": {
    "hypermnesia": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:8765/mcp",
               "--header", "Authorization: Bearer dev-token"]
    }
  }
}
```

### Claude API / Agent SDK

Pass the server via the MCP connector (the `mcp_servers` field), pointing at
`http://localhost:8765/mcp` with the `Authorization: Bearer <token>` header.

## Project scoping (no trampling)

Memories live in **scopes**, and the server derives each session's scope so a single
global config can't mix projects together:

1. **Workspace root** — MCP clients (Claude Code included) advertise the project
   directory as a root; the server maps it to `project:<dirname>-<hash>`. Saves and
   searches default to this scope automatically. No per-project setup.
2. **`X-Hypermnesia-Project` header** — override with a stable key (e.g. a repo slug) so
   a team or several machines share one project's memory. Set it per project in a
   project-scoped `.mcp.json`.
3. **`default`** — fallback when a client advertises neither.

`memory_search`/`memory_list` return the current project **plus** any granted shared
scopes (like `shared`) — never another project's. `memory_save` defaults to the project
scope; pass `scope: "shared"` to cross boundaries deliberately. Because the scope is
derived server-side from the real workspace, the model can't accidentally write to the
wrong project by mistyping a name.

> After changing the server's tool signatures, reconnect the MCP client (it caches the
> tool list on connect) to pick them up.

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

Run the whole suite against the Docker stack (dev overlay builds from source):

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.dev.yml \
  --profile test run --rm tests   # waits for health, runs unit + e2e
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
