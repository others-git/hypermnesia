"""Labelled recall corpus for the ranking eval.

The memories are modelled on a real agent memory store: long, prose-heavy notes
mixing prescriptive rules, incident write-ups and roadmaps. That shape matters —
the ranking bug this corpus exists to measure only appears when several notes
share incidental vocabulary while only one actually answers the query.

Each query names the memory that *should* rank first. `kind` splits the two
regimes hybrid search has to serve at once:

  conversational — natural-language questions, where the embedding carries the
                   meaning and lexical overlap is largely accidental.
  exact_token    — a rare identifier (error class, ticket id, env var, path),
                   where the embedding is weak and the lexical hit is the point.

A fix must hold both. Trading exact-token recall away to fix conversational
ranking would undo the reason hybrid search was added.
"""

from __future__ import annotations

# (key, description, content, tags, importance)
CORPUS: list[tuple[str, str, str, list[str], float]] = [
    (
        "test-coverage",
        "Workflow rule: every feature/change requires test coverage in the same commit",
        "Every feature, piece of functionality, or behavioural change must ship with "
        "test coverage in the same change — non-negotiable, not a follow-up. The user "
        "ships straight to main (no PR review gate), so tests are the only safety net. "
        "Add or extend unit and/or e2e tests in the same commit; if something is "
        "genuinely hard to test, say so explicitly rather than silently skipping. Run "
        "the suite before pushing.",
        ["workflow", "testing"],
        1.0,
    ),
    (
        "roadmap-todo",
        "TODO: remaining hypermnesia improvements — search_log retention, roots caching",
        "Remaining hypermnesia MCP-server improvements. SHIPPED: dedup content gate; "
        "recall metrics and memory_stats; scheduled forget sweep; admin CLI "
        "export/import/reindex. STILL TODO: search_log grows unbounded with no "
        "retention or pruning; _project_scope asks the client for roots on every tool "
        "call, so cache it per session; polish items are list pagination, batch save, "
        "and structured tool outputs. Long-term: Redis hot-cache, per-principal rate "
        "limits, a web UI. Process notes: ship straight to main, no PRs. Every change "
        "needs tests. Deploying needs the server container restarted because "
        "init_schema runs ALTERs at startup.",
        ["todo", "roadmap", "mcp-server"],
        1.0,
    ),
    (
        "privacy-identity",
        "Never include personal name/email in commits or code; use the GitHub identity only",
        "PRIVACY hard rule: never include the user's personal identity in anything "
        "committed, published, or sent anywhere. No real name, no personal email. Not "
        "in git author or committer identities, not in code such as user-agent or "
        "contact strings, not in docs or generated files. The only identity allowed in "
        "public artifacts is the GitHub one. Before the first commit in any repo, check "
        "git config user.name and user.email and set repo-local config if needed. "
        "Learned the hard way: eighteen commits plus a user-agent string had to be "
        "scrubbed with git-filter-repo and force-pushed.",
        ["privacy", "git", "identity"],
        2.0,
    ),
    (
        "concise-replies",
        "User wants concise chat replies; directive added to global CLAUDE.md",
        "Communication style: be concise. Default to short — the user does not read "
        "long answers. Lead with the result. No preamble, no restating the request. "
        "Report what changed and anything they must act on, cut everything else. Do not "
        "narrate process, list options considered, or summarise work they just watched "
        "happen. Prose for one or two points, a list only when it genuinely is a list. "
        "No section headers on a short answer. Still surface real problems, caveats and "
        "failures — brevity never means hiding bad news. Code comments and docs keep "
        "normal quality; this applies to chat replies only.",
        ["communication", "style", "preference"],
        2.0,
    ),
    (
        "fastmcp-roots",
        "FastMCP 4 removed Context.list_roots, silently collapsing all projects into the default scope",
        "GOTCHA: FastMCP 4.x removed Context.list_roots. The project scoping called it "
        "inside a bare except Exception, so the AttributeError was swallowed and every "
        "project silently fell back to the shared default scope — the exact "
        "cross-project trampling the scoping exists to prevent. Two causes: the API "
        "moved, so ctx.list_roots() is gone in 4.x but ctx.session.list_roots() still "
        "works; and the MCP roots capability is deprecated as of SEP-2577, so a server "
        "cannot request roots at all on a transport with no back-channel — you get "
        "NoBackChannelError: Cannot send 'roots/list'. The X-Hypermnesia-Project header "
        "is the mechanism that always works. Contributing factor: pyproject declared "
        "fastmcp>=2.3 with no upper bound, so a fresh install jumped 2.x to 4.0.3.",
        ["fastmcp", "mcp", "roots", "scoping", "gotcha"],
        1.8,
    ),
    (
        "list-payload",
        "MCP tools for agents: list/index responses must omit bodies",
        "DESIGN PRINCIPLE for MCP tools whose consumer is an agent: context is the "
        "scarce resource, so a tool's output shape is part of its API. Concrete case: "
        "memory_list documented itself as a cheap index but returned the full content "
        "of every memory. Measured on real data, content was 74% of the payload and "
        "about eleven times longer than the description that summarised it, so the "
        "default listing cost roughly 16k tokens instead of 4k. No test read content "
        "from a list result — pure dead weight. Fixed by omitting content by default "
        "with an opt-in full flag. Generalises: any browse tool an agent calls for "
        "orientation should return identifiers plus one-line summaries, and make "
        "bodies a second targeted call.",
        ["mcp", "tool-design", "context-budget"],
        1.7,
    ),
    (
        "relative-cutoff",
        "HM_SEARCH_RELATIVE_CUTOFF drops hits far below the best hit of the same search",
        "Recall precision work: an absolute similarity floor cannot separate signal "
        "from noise, because unrelated text scores well inside every embedding model's "
        "cosine range. Measured on the real store, the absolute floor at 0.4 admitted "
        "77 of 77 irrelevant memories. A relative gate — drop anything more than 0.15 "
        "below the best hit of the same search — cut that to 15 of 77 while keeping 11 "
        "of 11 relevant ones. End to end that took results per query from 8.0 down to "
        "1.5 with top-1 accuracy unchanged. Unlike an absolute floor it travels across "
        "embedding models. A lexical match bypasses it, as it does the floor.",
        ["recall", "precision", "config"],
        1.5,
    ),
    (
        "deploy-restart",
        "Deploying hypermnesia needs the server container restarted",
        "Deployment note: init_schema runs its ALTER statements at server startup, so "
        "a schema change does not take effect until the server container is actually "
        "restarted. Pulling a new image without restarting leaves the old process "
        "running against a half-migrated database. The compose stack is brought up "
        "with an overlay: the dev overlay builds from source, the release overlay "
        "pulls the published image.",
        ["deploy", "docker", "ops"],
        1.0,
    ),
    (
        "e2e-isolated-stack",
        "Run e2e tests in an isolated compose project without touching the live stack",
        "To run the end-to-end suite without disturbing the live server, use an "
        "isolated compose project name with the dev overlay and reset the published "
        "ports so nothing collides. Build, bring up the database and server, poll the "
        "health endpoint until it answers, then run the one-shot test profile, and "
        "finally tear the whole thing down including volumes. The live server "
        "generally runs an older image than main, so never point the suite at it.",
        ["testing", "docker", "e2e"],
        1.2,
    ),
    (
        "embedding-model-choice",
        "bge-small-en-v1.5 is the default embedder; its cosine range is compressed",
        "The default embedding model is BAAI/bge-small-en-v1.5 at 384 dimensions, run "
        "locally on CPU through fastembed. Its cosine range is compressed: unrelated "
        "text lands around 0.30 to 0.45 and genuinely relevant text around 0.55 and "
        "up. That is why the absolute similarity floor default is tuned to 0.4, and "
        "why that number must be re-tuned if the model changes. Changing models means "
        "a dimension change, which the admin reindex command exists to handle.",
        ["embeddings", "config", "tuning"],
        1.3,
    ),
    (
        "hnsw-ef-search",
        "Widen hnsw.ef_search past the candidate pool or filtered searches come back short",
        "pgvector's HNSW index post-filters on scope and tags during the index walk, "
        "so a filtered search can return fewer rows than the requested pool even when "
        "plenty of matching rows exist. The fix is to widen ef_search past the pool "
        "size transaction-locally before the query, capped at the pgvector maximum of "
        "1000. Without it, a narrow scope quietly loses candidates before ranking ever "
        "sees them.",
        ["pgvector", "hnsw", "tuning"],
        1.4,
    ),
    (
        "dedupe-content-gate",
        "Dedup needs a second content gate or descriptions cause false merges",
        "The save path merges a new memory into an existing one when their combined "
        "description-plus-content vectors are near-identical. On its own that produced "
        "false merges: two genuinely different notes with similarly worded "
        "descriptions were collapsed into one, silently losing the older content. The "
        "second gate compares the raw contents directly rather than the stored "
        "combined vectors, so a description-driven false positive cannot survive it. "
        "The threshold is configurable and setting it to zero disables the gate.",
        ["dedup", "data-loss", "save"],
        1.6,
    ),
    (
        "forget-importance",
        "Default-importance memories become eligible for the forget sweep once stale",
        "The forget sweep archives memories that have not been recalled within the "
        "retention window and whose importance is at or below the floor. Both "
        "conditions must hold. Because the default importance and the default floor "
        "are the same value, an ordinary memory becomes eligible as soon as it goes "
        "stale — anything that must survive needs its importance raised above the "
        "floor. Archiving is a soft delete and can be undone.",
        ["forget", "retention", "gotcha"],
        1.5,
    ),
    (
        "async-embed-offload",
        "Embedders are synchronous, so they must run in a worker thread",
        "The embedding providers are all synchronous — either CPU-bound ONNX "
        "inference or a blocking HTTP call. Calling one directly from a coroutine "
        "stalls the event loop, so a single embed blocks every concurrent request. "
        "Every embed call is offloaded to a worker thread instead. This was a real "
        "latency problem under concurrent load, not a theoretical one.",
        ["async", "performance", "embeddings"],
        1.3,
    ),
    (
        "admin-cli-reindex",
        "Admin CLI does JSONL export/import and reindex for embedding-model migration",
        "The admin command line tool exists for operations the MCP surface should not "
        "expose. Export writes every memory to JSON lines including its metadata; "
        "import reads them back. Reindex re-embeds the whole store with a different "
        "model, which is what makes a dimension change survivable — it rewrites the "
        "vector column and rebuilds the index rather than requiring a wipe.",
        ["admin", "cli", "migration"],
        1.2,
    ),
    (
        "search-log-unbounded",
        "search_log has no retention policy and grows without bound",
        "Every search writes one row to the search log table, which is what makes the "
        "stats tool possible. There is no pruning or retention policy on it, so the "
        "table grows without bound. At current volume that is only a few dozen rows a "
        "year, so it is low urgency, but it is a genuinely unbounded table and should "
        "eventually get a retention window.",
        ["observability", "tech-debt"],
        1.0,
    ),
    (
        "no-real-name-placeholder",
        "Use a dev-test placeholder identity in examples, never the user's name",
        "In example configuration, seeded fixtures, and documentation, the principal "
        "and scope names must use a neutral placeholder rather than anything derived "
        "from the user's real name. This shows up in the auth token map in the compose "
        "file and in the env example, both of which are committed.",
        ["privacy", "docs"],
        1.8,
    ),
    (
        "ship-to-main",
        "Ship straight to main; there is no pull request review gate",
        "Process: changes go straight to main. There is no pull request and no review "
        "gate, so nothing catches a regression except the test suite and whatever "
        "checking happens before the push. Branch protection is deliberately off. This "
        "is why test coverage is treated as non-negotiable rather than as a nice to "
        "have.",
        ["workflow", "git", "process"],
        1.4,
    ),
    (
        "auth-token-scopes",
        "Bearer tokens map to a principal and the scopes it may read and write",
        "Authentication is a JSON map from bearer token to a principal name and the "
        "list of scopes that principal may read and write. A request carrying an "
        "unknown token is rejected outright. Requiring auth can be turned off for "
        "local development, but the shipped default requires it, and the example "
        "tokens must be replaced before the server is exposed anywhere.",
        ["auth", "security", "config"],
        1.4,
    ),
    (
        "recency-half-life",
        "Ranking blends similarity with a recency decay and importance",
        "The final score is a weighted blend of a relevance signal, an exponential "
        "recency decay measured from when the memory was last recalled, and the "
        "memory's importance normalised against a cap. Similarity carries the largest "
        "weight so relevance leads, with recency and importance acting as tie "
        "breakers within a neighbourhood rather than as primary sort keys. Recalling a "
        "memory refreshes its recency, which also shields it from being forgotten.",
        ["ranking", "scoring", "design"],
        1.5,
    ),
]

# (query, expected_top_key, kind)
QUERIES: list[tuple[str, str, str]] = [
    # --- conversational: meaning carries the query, lexical overlap is incidental ---
    ("do I need tests for this change", "test-coverage", "conversational"),
    ("should I write tests before shipping", "test-coverage", "conversational"),
    ("how long should my replies be", "concise-replies", "conversational"),
    ("can I put the user's email in a commit", "privacy-identity", "conversational"),
    ("what is left to do on this project", "roadmap-todo", "conversational"),
    ("why did every project share the same memories", "fastmcp-roots", "conversational"),
    ("how do I stop irrelevant results burying the good one", "relative-cutoff", "conversational"),
    ("do I need to restart anything after deploying", "deploy-restart", "conversational"),
    ("how do I test without breaking the live server", "e2e-isolated-stack", "conversational"),
    ("why are unrelated memories scoring so high", "embedding-model-choice", "conversational"),
    ("my filtered search returns fewer rows than I asked for", "hnsw-ef-search", "conversational"),
    ("two different notes got merged into one", "dedupe-content-gate", "conversational"),
    ("why did my memory disappear", "forget-importance", "conversational"),
    ("requests are slow when several run at once", "async-embed-offload", "conversational"),
    ("how do I change the embedding model", "admin-cli-reindex", "conversational"),
    ("does anything clean up old rows", "search-log-unbounded", "conversational"),
    ("do we review changes before merging", "ship-to-main", "conversational"),
    ("how does a client prove who it is", "auth-token-scopes", "conversational"),
    ("what makes one result rank above another", "recency-half-life", "conversational"),
    ("should list responses include the whole body", "list-payload", "conversational"),
    # --- exact_token: a rare identifier the embedding handles poorly ---
    ("SEP-2577", "fastmcp-roots", "exact_token"),
    ("NoBackChannelError", "fastmcp-roots", "exact_token"),
    ("Context.list_roots", "fastmcp-roots", "exact_token"),
    ("HM_SEARCH_RELATIVE_CUTOFF", "relative-cutoff", "exact_token"),
    ("hnsw.ef_search", "hnsw-ef-search", "exact_token"),
    ("bge-small-en-v1.5", "embedding-model-choice", "exact_token"),
    ("git-filter-repo", "privacy-identity", "exact_token"),
    ("init_schema", "deploy-restart", "exact_token"),
    ("reindex", "admin-cli-reindex", "exact_token"),
]
