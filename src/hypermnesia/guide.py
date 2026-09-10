"""Self-documenting usage guide, generated from the live server.

The problem this solves: the guidance that tells an agent *when* to call these
tools has to live in the client's `CLAUDE.md`, far away from the code. Every
hand-maintained copy of it drifts — ours was still describing tool signatures
and defaults that had since changed. So the server generates it instead, from
its own settings and registered tools, and the agent writes the result.

Everything here is a pure function of its arguments so the whole guide can be
tested without a server, a database, or a client.
"""

from __future__ import annotations

import textwrap
from typing import Any, Iterable

from .config import Settings

# A tool is worth a line in the guide only if the agent needs to know *when* to
# reach for it; the tool's own description covers *how*. Anything not listed
# here still shows up under "other tools", so a new tool can never go missing.
_WHEN_TO_USE = {
    "memory_search": "at the start of a task, to recall relevant prior context",
    "memory_save": "when you learn a durable fact, preference, decision, or gotcha",
    "memory_update": "to fix or extend a memory you already have the id for",
    "memory_list": "to see what exists, as a cheap index (no bodies)",
    "memory_get": "to read one memory's full body, after finding it in the index",
    "memory_forget": "housekeeping: archive stale, low-importance clutter",
    "memory_restore": "to bring back something forgetting took away",
    "memory_delete": "the hard, irreversible removal",
    "memory_stats": "when recall feels off, to check its health from evidence",
    "memory_usage_guide": "to regenerate this guidance after a server upgrade",
}


def tool_summaries(objs: Iterable[Any]) -> list[tuple[str, str]]:
    """``(name, first docstring line)`` for each registered tool.

    Accepts whatever ``@mcp.tool`` returned: FastMCP 3 hands back the plain
    function, FastMCP 4 a wrapper object. Reading both shapes keeps the guide
    generated rather than hand-listed, which is the entire point of it.
    """
    out: list[tuple[str, str]] = []
    for obj in objs:
        # A module's globals hold plenty of things with a `.name` that isn't a
        # tool name (or even a string), so check the type before trusting it.
        name = getattr(obj, "name", None)
        if not isinstance(name, str):
            name = getattr(obj, "__name__", None)
        if not isinstance(name, str) or not name.startswith("memory_"):
            continue
        doc = getattr(obj, "description", None)
        if not doc:
            fn = getattr(obj, "fn", obj)
            doc = getattr(fn, "__doc__", "") or ""
        first = next((ln.strip() for ln in doc.strip().splitlines() if ln.strip()), "")
        out.append((name, first))
    return sorted(out)


_BIND_ALL = ("0.0.0.0", "127.0.0.1", "::", "")


def _server_url(settings: Settings) -> tuple[str, bool]:
    """The URL clients should dial, and whether it had to be guessed.

    ``HM_HOST`` is a *bind* address: "0.0.0.0" means "every interface", which is
    not somewhere a client can connect to, and the server has no way to know
    which of its addresses a given client can reach. So substitute localhost and
    admit the guess rather than printing a URL that may simply be wrong.
    """
    guessed = settings.host in _BIND_ALL
    host = "localhost" if guessed else settings.host
    return f"http://{host}:{settings.port}/mcp", guessed


def session_report(
    principal_id: str,
    project_scope: str,
    read_scopes: list[str],
    roots_resolved: bool,
) -> dict[str, Any]:
    """What this specific session actually resolved to.

    The guide is worth more than a static doc precisely because it can say
    "here is what *your* connection is doing", including the parts that are
    silently degraded.
    """
    return {
        "principal": principal_id,
        "project_scope": project_scope,
        "scopes_you_can_read": read_scopes,
        "saves_default_to": project_scope,
        "workspace_roots_detected": roots_resolved,
        "project_isolation": (
            "active" if project_scope != "default" else "NOT ACTIVE — see warnings"
        ),
    }


def warnings_for(project_scope: str, roots_resolved: bool, settings: Settings) -> list[str]:
    """Setup problems this session can actually observe, phrased as fixes."""
    out: list[str] = []
    if project_scope == "default":
        out.append(
            "This session resolved to the 'default' scope, which every project "
            "using this server shares — your memories are NOT isolated per "
            "project. Fix: set the X-Hypermnesia-Project header per project "
            "(see the 'setup' section). Workspace roots would do it "
            "automatically, but the MCP roots capability is deprecated "
            "(SEP-2577) and unavailable on transports without a back-channel, "
            "so the header is the reliable mechanism."
        )
    elif not roots_resolved:
        out.append(
            "Workspace roots were not delivered by this client, so the project "
            "scope came from the X-Hypermnesia-Project header. That is the "
            "supported setup — no action needed."
        )
    if not settings.require_auth:
        out.append(
            "HM_REQUIRE_AUTH is false: every caller is 'anonymous' with access "
            "to the 'default' scope. Do not expose this server on a network."
        )
    return out


def setup_commands(settings: Settings, project_key: str = "my-project") -> str:
    """Client registration snippets, with this server's address."""
    url, guessed = _server_url(settings)
    note = (
        f"\n> The server binds `{settings.host}`, which is a bind address rather "
        f"than a route, so `{url}` above is a guess. If you reach this server "
        "over the network, substitute the host or IP your client actually "
        "connects to.\n"
        if guessed
        else ""
    )
    return f"""\
### Register the server (Claude Code)

```bash
claude mcp add --transport http hypermnesia {url} \\
  --header "Authorization: Bearer <your-token>" \\
  --scope user
```

### Keep projects apart (recommended)

Project scoping is what stops one project's memories leaking into another. Set a
stable key per project in that project's `.mcp.json`:

```json
{{
  "mcpServers": {{
    "hypermnesia": {{
      "type": "http",
      "url": "{url}",
      "headers": {{
        "Authorization": "Bearer <your-token>",
        "X-Hypermnesia-Project": "{project_key}"
      }}
    }}
  }}
}}
```

### Claude Desktop

`claude_desktop_config.json` is stdio-only, so bridge with `mcp-remote`:

```json
{{
  "mcpServers": {{
    "hypermnesia": {{
      "command": "npx",
      "args": ["mcp-remote", "{url}",
               "--header", "Authorization: Bearer <your-token>",
               "--header", "X-Hypermnesia-Project: {project_key}"]
    }}
  }}
}}
```

> Replace `<your-token>` with a token from `HM_AUTH_TOKENS`. This guide never
> prints real tokens.
{note}"""


def _wrap(blocks: list[str], width: int = 79) -> str:
    """Wrap prose to a fixed width, leaving structure alone.

    Values interpolated from settings vary in length, so hardcoding the line
    breaks around them produces ragged text the moment a default changes.
    """
    out: list[str] = []
    for b in blocks:
        if not b or b.startswith(("#", "```")):
            out.append(b)
        elif b.startswith("- "):
            out.append(textwrap.fill(b, width, subsequent_indent="  "))
        else:
            out.append(textwrap.fill(b, width))
    return "\n".join(out).rstrip() + "\n"


def claude_md_block(
    settings: Settings,
    tools: list[tuple[str, str]],
    project_scope: str = "",
) -> str:
    """The guidance section to paste into a `CLAUDE.md`.

    Defaults are interpolated from the live settings, so the instructions
    describe how this server actually behaves rather than how it behaved when
    someone last edited the README.
    """
    s = settings
    by_name = dict(tools)
    # Workflow order (the order of _WHEN_TO_USE), not alphabetical: the list is
    # guidance about when to reach for what, so it should read like the flow.
    known = [(n, by_name[n]) for n in _WHEN_TO_USE if n in by_name]
    other = [n for n, _ in tools if n not in _WHEN_TO_USE]

    blocks = [
        "## Persistent memory (hypermnesia MCP)",
        "",
        "A semantic memory store that persists across sessions. Use it instead of "
        "assuming context carries over.",
        "",
        "**Recall.** At the start of a non-trivial task, call `memory_search` with a "
        "short query describing what you're about to do. Recall is hybrid (semantic "
        "+ keyword), so exact tokens \u2014 error codes, flag names, paths \u2014 are found "
        "too. Each hit carries a raw `similarity` and a blended `score` (relevance + "
        "recency + importance).",
        "",
    ]

    if s.search_relative_cutoff > 0:
        blocks += [
            f"Results are trimmed to hits within {s.search_relative_cutoff} similarity "
            "of the best one, so you get the answer rather than everything adjacent "
            "to it. For a deliberately broad survey pass `relative_cutoff: 0.0` "
            "(together with `min_similarity: 0.0`).",
            "",
        ]
    else:
        blocks += [
            "The relative cutoff is disabled on this server, so a search returns "
            f"everything above the {s.search_min_similarity} similarity floor.",
            "",
        ]

    blocks += [
        "**Saving.** When you learn a durable fact, preference, decision, or "
        "hard-won gotcha that will matter in a future session, call `memory_save` "
        "with a clear one-line `description` and the `content`. Set `type` to one of "
        "`fact | preference | project | reference`.",
        "",
        (
            f"- Don't pass `scope`: it defaults to this project (`{project_scope}`)."
            if project_scope
            else "- Don't pass `scope`: it defaults to the current project."
        ),
        '- Pass `scope: "shared"` only for things useful across every project.',
        f"- `importance` defaults to 1.0 (cap {s.importance_cap}). Raise genuinely "
        "durable facts to ~1.5-2.0: it ranks them up and, above the forget floor of "
        f"{s.forget_importance_floor}, protects them from being forgotten.",
        f"- A near-duplicate in the same scope (combined embedding >= "
        f"{s.dedupe_threshold} and contents agreeing at >= "
        f"{s.dedupe_content_threshold}) is updated in place: the result has "
        "`created: false` and a `replaced` block holding the pre-merge memory. "
        "Glance at it to catch a wrong merge.",
        "",
        "**Browsing.** `memory_list` is an index \u2014 descriptions and metadata, no "
        "bodies. Scan it to see what exists, then `memory_get(memory_id)` for the "
        "one you need. Use `full: true` only when you really want every body.",
        "",
        "**Editing.** Prefer `memory_update(memory_id, ...)` over re-saving when you "
        "already know the id; only the fields you pass change.",
        "",
        "**Housekeeping.** `memory_forget` archives memories not recalled in "
        f"{s.forget_after_days:g} days whose importance is at or below "
        f"{s.forget_importance_floor} \u2014 a dry run unless you pass `apply: true`. "
        "`memory_list(include_archived=true)` reviews them and `memory_restore` "
        "brings one back; `memory_delete` is the hard removal. `memory_stats` "
        "reports recall health (volume, empty-result rate, recent misses) when "
        "recall feels off.",
        "",
        "**Hygiene.** Search before saving, and prefer updating a near-duplicate "
        "over creating a new memory. Save the non-obvious \u2014 preferences, "
        "conventions, decisions \u2014 not what the repo, git history, or its own files "
        "already record.",
        "",
        "**Tools**",
        "",
    ]
    for name, desc in known:
        blocks.append(f"- `{name}` \u2014 {_WHEN_TO_USE[name]}. {desc}")
    if other:
        blocks += [
            "",
            "Other tools on this server: " + ", ".join(f"`{n}`" for n in other) + ".",
        ]
    return _wrap(blocks)


def build(
    *,
    settings: Settings,
    tools: list[tuple[str, str]],
    principal_id: str,
    project_scope: str,
    read_scopes: list[str],
    roots_resolved: bool,
    section: str = "all",
) -> dict[str, Any]:
    """Assemble the requested sections of the guide."""
    valid = {"all", "claude_md", "setup", "session"}
    if section not in valid:
        raise ValueError(f"unknown section {section!r}; expected one of {sorted(valid)}")

    # A real project key beats the placeholder in the setup snippets.
    key = project_scope[len("project:"):] if project_scope.startswith("project:") else "my-project"

    out: dict[str, Any] = {
        "warnings": warnings_for(project_scope, roots_resolved, settings),
    }
    if section in ("all", "claude_md"):
        out["claude_md"] = claude_md_block(settings, tools, project_scope)
    if section in ("all", "setup"):
        out["setup"] = setup_commands(settings, key)
    if section in ("all", "session"):
        out["session"] = session_report(
            principal_id, project_scope, read_scopes, roots_resolved
        )
    return out
