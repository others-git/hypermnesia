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
| `memory_search(query, scope?, tags?, k, min_similarity?, relative_cutoff?)` | Hybrid (semantic + keyword) recall; ranks by relevance + recency + importance |
| `memory_save(content, description, scope, type?, tags?, metadata?, importance?)` | Store; updates a near-duplicate instead of inserting |
| `memory_update(memory_id, content?, description?, type?, tags?, metadata?, importance?)` | Edit a known memory in place (only given fields change) |
| `memory_get(memory_id)` | Fetch one by id |
| `memory_list(scope?, tags?, limit, include_archived?, full?)` | Cheap index: descriptions + metadata, no bodies (`full: true` for bodies); `include_archived` to review forgotten ones |
| `memory_delete(memory_id)` | Delete by id (hard) |
| `memory_forget(scope?, tags?, older_than_days?, importance_floor?, apply?, limit?)` | Archive stale, low-importance memories; dry-run unless `apply: true` |
| `memory_restore(memory_id)` | Un-archive a forgotten memory (inverse of `memory_forget`) |
| `memory_stats(scope?, days?)` | Recall health: store size per scope + search volume, empty-rate, score/latency metrics |
| `memory_usage_guide(section?)` | Generates setup + `CLAUDE.md` guidance from this server's live settings and tools |

Search results carry both a raw `similarity` (0-1 cosine) and a blended `score` that
adds recency decay (half-life `HM_RECENCY_HALF_LIFE_DAYS`) and normalised `importance`;
tune the mix via `HM_SCORE_WEIGHT_*`.

Two gates keep weak matches out of the agent's context, and they do different jobs:

- **`HM_SEARCH_MIN_SIMILARITY`** (default `0.4`) is an *absolute* floor — it answers
  "is anything here relevant at all?". It is tied to the model's cosine scale
  (bge-small-en-v1.5: unrelated text ~0.30-0.45, relevant ~0.55+), so re-tune it if
  you switch embedding models.
- **`HM_SEARCH_RELATIVE_CUTOFF`** (default `0.15`) is a *relative* one — it drops hits
  more than that far below the best hit of the same search, answering "is this hit
  worth reading next to the best one?". An absolute floor cannot do this job: on a
  measured 11-query eval, the 0.4 floor admitted **every** irrelevant memory (77/77),
  and the tightest floor that trimmed a comparable amount had already started
  discarding relevant hits. The relative gate cut irrelevant hits to 15/77 while
  keeping all 11 relevant ones. End to end on a 10-memory store it took recall from
  8.0 to 1.5 results per query with no change in top-1 accuracy.

Both are overridable per search (`min_similarity`, `relative_cutoff`), and `0.0`
disables either. Note that `min_similarity: 0.0` alone no longer means "return
everything" — pass `relative_cutoff: 0.0` too for a broad survey. The relative gate
deliberately does nothing when results are flat (an off-topic query with no standout),
so it narrows a good answer's neighbourhood rather than inventing one, and lexical
hits bypass it exactly as they bypass the floor.

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
so the next sweep won't immediately re-forget it. In both dry-run and apply results,
`matched` is the true total the criteria hit (exactly what apply archives) while
`memories` lists at most `limit` (default 100) of them, stalest first, with
`truncated` set when the list was cut — a dry run can never under-report an apply.
Set `HM_FORGET_SWEEP_HOURS` (default 0 = off) to have the server run the sweep itself
periodically over every scope, using the two thresholds above; archived descriptions
are logged each pass.

**Observability.** Every `memory_search` is logged (query, candidate/hit counts, top
scores, latency) to a `search_log` table, and `memory_stats` aggregates it: search
volume, empty-result rate, average hits and top score, latency percentiles, and the
most recent queries that returned nothing — the recall misses worth investigating.
Use it to tune `HM_SEARCH_MIN_SIMILARITY` and the score weights from evidence instead
of feel. Logged queries land in the DB; disable with `HM_SEARCH_LOG_ENABLED=false`.

`description` is a one-line summary used for ranking and de-duplication — treat it like
the one-liners in Claude Code's `MEMORY.md` index. It is also what makes `memory_list`
an index worth calling: it returns descriptions and metadata but not bodies, because
content runs ~11x longer than its description and would otherwise be ~74% of the
payload (a default `limit=50` listing costs ~4k tokens instead of ~16k). Scan the
descriptions, then `memory_get` the one you need; `full: true` returns bodies when you
really do want them all. De-duplication is two-gated: the
combined embedding must clear `HM_DEDUPE_THRESHOLD` **and** the contents themselves must
agree (`HM_DEDUPE_CONTENT_THRESHOLD`, default 0.9) — so two different facts that merely
share a description shape insert side by side instead of one silently clobbering the
other. Failing the gate errs toward a duplicate (recoverable) over a lost fact (not).

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

Exposing tools isn't enough — Claude won't reach for them unless told when to. Rather
than copying a block from this README (which drifts the moment a default or signature
changes), ask the server for it:

> Call `memory_usage_guide` and write the `claude_md` section into my `CLAUDE.md`.

`memory_usage_guide` generates the guidance from the server's **live** settings and
registered tool list, so it always states how your server actually behaves — the real
similarity cutoff, forget thresholds, dedup gates, and every tool currently exposed. It
returns four things:

| Key | What it is |
|---|---|
| `claude_md` | A ready-to-write `CLAUDE.md` section: when to call each tool, with live defaults |
| `setup` | Client registration snippets using this server's real address (never real tokens) |
| `session` | What your connection resolved to: principal, project scope, readable scopes, whether roots were detected |
| `warnings` | Setup problems the server can actually observe — above all, project isolation not being active |

Pass `section` (`claude_md` \| `setup` \| `session`) to get just one; `warnings` always
rides along. Re-run it after upgrading the server to refresh a stale `CLAUDE.md`.

**Check `warnings` on first setup.** If your client doesn't deliver workspace roots and
you haven't set the `X-Hypermnesia-Project` header, every project shares the `default`
scope — the guide says so explicitly and hands you the `.mcp.json` that fixes it.

You can put the generated section in a **single global** `~/.claude/CLAUDE.md` — memories
are partitioned per project (see below), so projects never trample each other.

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

1. **`X-Hypermnesia-Project` header** — a stable key (e.g. a repo slug), mapped to
   `project:<slug>`. Set it per project in a project-scoped `.mcp.json`. This is the
   mechanism that works on every client and transport, and it also lets a team or
   several machines share one project's memory deliberately.
2. **Workspace root** — MCP clients (Claude Code included) advertise the project
   directory as a root; the server maps it to `project:<dirname>-<hash>` with no
   per-project setup. This is **best effort**: the MCP roots capability is deprecated
   (SEP-2577), and a server cannot request roots at all over a transport with no
   back-channel for server-initiated requests. Where it works it is the most
   convenient option; where it doesn't, the server falls back to step 3.
3. **`default`** — fallback when neither is available. Every project sharing this
   server lands in one pool here, so the server logs a warning (once per process) the
   first time a request falls back. **If you see that warning, set the header** — it
   means projects are no longer isolated.

`memory_search`/`memory_list` return the current project **plus** any granted shared
scopes (like `shared`) — never another project's. `memory_save` defaults to the project
scope; pass `scope: "shared"` to cross boundaries deliberately. Because the scope is
derived server-side from the session, the model can't accidentally write to the wrong
project by mistyping a name.

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

- **Unit** (`test_auth`, `test_embeddings`, `test_scoping`, `test_fuse`, `test_roots`,
  `test_dedupe_gate`, `test_embed_offload`, `test_sweep`, `test_admin`, `test_guide`):
  pure logic, no
  infra — `pytest -m "not e2e"`.
- **End-to-end** (`tests/test_e2e.py`): black-box CRUD + semantic recall + auth/scope
  isolation, driven through the MCP tools against a running server. They auto-skip if no
  server is reachable.
- **Admin CLI e2e** (`tests/test_cli_e2e.py`): export/import restore + reindex model
  swap. Need direct DB access, so they only run where `HM_TEST_DATABASE_URL` is set
  (the compose test container sets it); `HM_TEST_ALT_MODEL` gates the
  dimension-change reindex test.

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
| `HM_SEARCH_MIN_SIMILARITY` | `0.4` | absolute cosine floor; model-specific, re-tune on a model swap |
| `HM_SEARCH_RELATIVE_CUTOFF` | `0.15` | drop hits this far below the best hit; `0` disables |
| `HM_HYBRID_SEARCH` | `true` | fuse vector + Postgres full-text recall (`HM_RRF_K`, `HM_HYBRID_*_WEIGHT`) |
| `HM_SCORE_WEIGHT_*` | `1.0`/`0.25`/`0.15` | similarity / recency / importance blend |
| `HM_DEDUPE_THRESHOLD` | `0.92` | cosine sim above which `save` updates vs. inserts |
| `HM_DEDUPE_CONTENT_THRESHOLD` | `0.9` | second gate: contents must agree too, or it inserts |
| `HM_FORGET_AFTER_DAYS` | `180` | staleness threshold for `memory_forget` |
| `HM_FORGET_IMPORTANCE_FLOOR` | `1.0` | only memories at or below this are forgettable |
| `HM_FORGET_SWEEP_HOURS` | `0` | server-side periodic sweep; `0` disables |
| `HM_SEARCH_LOG_ENABLED` | `true` | log searches to `search_log` for `memory_stats` |
| `HM_LOG_LEVEL` | `INFO` | `DEBUG` surfaces roots-handshake and scoping detail |
| `HM_AUTH_TOKENS` | `{}` | `{"token":{"principal":"id","scopes":["..."]}}` |
| `HM_REQUIRE_AUTH` | `true` | when false, all callers are `anonymous`/`default` |

## Admin CLI (backup & model migration)

The `hypermnesia` binary doubles as an admin CLI. These subcommands run on the server
host with **direct DB access** (they read `HM_DATABASE_URL` etc.) and bypass MCP
auth/scopes on purpose — they are the backup and migration story, not agent tools:

```bash
hypermnesia export --out dump.jsonl        # all memories (archived too), as JSONL
hypermnesia export --scope shared          # limit to scopes; repeatable; - = stdout
hypermnesia import dump.jsonl              # restore a dump; existing ids are skipped
hypermnesia reindex                        # re-embed everything with HM_EMBEDDING_*
```

Dumps exclude embeddings, so they survive an embedding-model change — `import`
re-embeds with the configured model, preserving ids and timestamps and never
overwriting an existing row (re-running a restore is idempotent). In a Docker setup:
`docker compose exec server hypermnesia export --out - > dump.jsonl`. **Take dumps
periodically** — the Postgres volume is otherwise the only copy of every memory.

### Swapping the embedding model

Set `HM_EMBEDDING_PROVIDER` / `HM_EMBEDDING_MODEL`. The vector dimension is auto-detected
and **pinned** in the store on first run. Starting the server with a different model
(or dimension) is refused with a clear error, because existing vectors would no longer
be comparable. The migration path is `reindex`:

```bash
HM_EMBEDDING_MODEL=new-model hypermnesia reindex   # re-embeds all rows, swaps the
                                                   # vector column + HNSW index, and
                                                   # updates the pin — one transaction
# then restart the server with the new HM_EMBEDDING_* settings
```

Until the restart, the running server still embeds with the old model and will error —
run reindex during a quiet moment. Also re-tune `HM_SEARCH_MIN_SIMILARITY` after a
model swap (cosine ranges differ per model).

To add a new provider, implement the `Embedder` protocol and `@register("name")` it in
`src/hypermnesia/embeddings/providers.py`.

## Status

**v1.** Implemented: MCP tools, pgvector storage, scope-based auth/isolation, hybrid
(semantic + keyword) recall with absolute and relative precision gates, two-gated
de-duplication, forgetting (manual + scheduled sweep) with restore, search logging and
`memory_stats`, an admin CLI for backup and model migration, and a unit + e2e suite.

Roadmap: Redis hot-cache, per-principal rate limits, `search_log` retention (the table
currently grows unbounded), caching workspace roots per session instead of re-asking
per tool call, web UI.
