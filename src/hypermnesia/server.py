from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers

from .auth import (
    AuthError,
    effective_read_scopes,
    effective_write_scope,
    resolve_principal,
)
from .config import Principal, get_settings
from .guide import build as build_guide
from .guide import tool_summaries
from .scoping import derive_project_scope
from .service import MemoryService

mcp = FastMCP("hypermnesia")

logger = logging.getLogger("hypermnesia")

_service: MemoryService | None = None
_warned_no_project = False
_sweep_task: asyncio.Task | None = None
_lock = asyncio.Lock()


async def run_forget_sweep_once(svc: MemoryService, settings) -> dict[str, Any]:
    """One pass of the periodic forget sweep, over every scope in the store."""
    result = await svc.forget(
        await svc.distinct_scopes(),
        older_than_days=settings.forget_after_days,
        importance_floor=settings.forget_importance_floor,
        apply=True,
    )
    if result["matched"]:
        logger.info(
            "forget sweep archived %d stale memories: %s",
            result["matched"],
            [m["description"] for m in result["memories"]],
        )
    return result


async def forget_sweep_loop(svc: MemoryService, settings) -> None:
    """Sleep-first loop (no surprise archiving at boot); one failed pass is
    logged and the loop keeps going."""
    while True:
        await asyncio.sleep(settings.forget_sweep_hours * 3600)
        try:
            await run_forget_sweep_once(svc, settings)
        except Exception:  # noqa: BLE001 - the sweep must outlive one bad pass
            logger.exception("forget sweep failed")


async def _get_service() -> MemoryService:
    global _service, _sweep_task
    if _service is None:
        async with _lock:
            if _service is None:
                settings = get_settings()
                _service = await MemoryService.create(settings)
                if settings.forget_sweep_hours > 0:
                    _sweep_task = asyncio.create_task(
                        forget_sweep_loop(_service, settings)
                    )
    return _service


def _principal() -> Principal:
    # include_all=True so the Authorization header isn't filtered out.
    headers = get_http_headers(include_all=True)
    try:
        return resolve_principal(get_settings(), headers.get("authorization"))
    except AuthError as e:
        raise ToolError(str(e)) from e


async def _list_roots(ctx: Context) -> list[str]:
    """Ask the client for its workspace roots, across FastMCP versions.

    FastMCP 4 dropped ``Context.list_roots``; the MCP session method underneath
    it stayed. Trying both keeps project scoping working on 2.x through 4.x --
    when it broke, every project silently shared the ``default`` scope, which is
    exactly the trampling this scoping is here to prevent.
    """
    candidates = (
        getattr(ctx, "list_roots", None),
        getattr(getattr(ctx, "session", None), "list_roots", None),
    )
    for call in candidates:
        if call is None:
            continue
        try:
            result = await call()
        except Exception as e:  # noqa: BLE001 - a client may not support roots
            # Debug, not warning: a client that genuinely advertises no roots
            # takes this path on every call and is a supported configuration.
            logger.debug("roots request failed (%s: %s)", type(e).__name__, e)
            continue
        raw = getattr(result, "roots", result)
        return [str(getattr(r, "uri", r)) for r in raw]
    logger.debug("no usable roots API on this FastMCP version; falling back")
    return []


async def _project_scope(ctx: Context) -> str:
    """Derive the calling session's project scope from its workspace root.

    Clients (e.g. Claude Code) advertise the project directory as an MCP root;
    a ``X-Hypermnesia-Project`` header overrides it for stable/shared keys.

    Roots are best-effort: the capability is deprecated (SEP-2577) and a server
    cannot request them at all on a transport with no back-channel. The header
    is the mechanism that always works.
    """
    scope, _ = await _project_scope_with_roots(ctx)
    return scope


async def _project_scope_with_roots(ctx: Context) -> tuple[str, bool]:
    """As ``_project_scope``, also reporting whether roots actually resolved.

    The guide reports on the session's real configuration, and "roots worked"
    is exactly the fact that was silently wrong before.
    """
    global _warned_no_project
    headers = get_http_headers(include_all=True)
    override = headers.get("x-hypermnesia-project")
    roots = await _list_roots(ctx)
    scope = derive_project_scope(roots, override)
    if scope == "default" and not _warned_no_project:
        # Once per process, at warning level: this is the state where every
        # project shares one pool of memories, and it is indistinguishable from
        # working until two projects have already contaminated each other.
        _warned_no_project = True
        logger.warning(
            "no workspace roots and no X-Hypermnesia-Project header: memories "
            "fall back to the 'default' scope, shared by every project using "
            "this server. Set the X-Hypermnesia-Project header per project to "
            "keep them apart."
        )
    return scope, bool(roots)


def _read_scopes(p: Principal, project_scope: str, requested: str | None) -> list[str]:
    try:
        return effective_read_scopes(p, project_scope, requested)
    except AuthError as e:
        raise ToolError(str(e)) from e


@mcp.tool
async def memory_search(
    ctx: Context,
    query: str,
    scope: str | None = None,
    tags: list[str] | None = None,
    k: int = 8,
    min_similarity: float | None = None,
    relative_cutoff: float | None = None,
) -> list[dict[str, Any]]:
    """Semantically recall memories relevant to `query`.

    Search this before starting a task. By default searches the current project's
    memories plus any shared scopes — never another project's. Results are ranked
    by a blend of semantic `similarity` (0-1), recency, and importance, exposed as
    `score`.

    Two knobs trade recall for precision. `min_similarity` (e.g. 0.3) is an
    absolute floor on `similarity`. `relative_cutoff` (default 0.15) drops hits
    that fall more than that far below the best hit of this search — it is what
    keeps a good answer from arriving buried in near-miss memories. Raise it (or
    pass 0.0 to disable) when you want a broad survey of everything related
    rather than the best answer.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), scope)
    svc = await _get_service()
    hits = await svc.search(
        query=query, scopes=scopes, tags=tags, k=k, min_similarity=min_similarity,
        relative_cutoff=relative_cutoff, owner_id=p.id,
    )
    return [h.model_dump(mode="json") for h in hits]


@mcp.tool
async def memory_save(
    ctx: Context,
    content: str,
    description: str,
    scope: str | None = None,
    type: str = "fact",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    importance: float = 1.0,
) -> dict[str, Any]:
    """Store a memory. `description` is a one-line summary used for ranking/dedup.

    By default the memory is scoped to the current project. Pass `scope: "shared"`
    to store something useful across projects. If a near-duplicate already exists
    in the target scope it is updated instead of inserted (`created: false`), and
    `replaced` holds what that memory looked like before the merge — check it to
    catch a wrong overwrite. `type` is one of fact | preference | project | reference.
    """
    p = _principal()
    try:
        scope = effective_write_scope(p, await _project_scope(ctx), scope)
    except AuthError as e:
        raise ToolError(str(e)) from e
    svc = await _get_service()
    memory, created, replaced = await svc.save(
        owner_id=p.id,
        scope=scope,
        content=content,
        description=description,
        type=type,
        tags=tags,
        metadata=metadata,
        importance=importance,
    )
    out: dict[str, Any] = {"created": created, "memory": memory.model_dump(mode="json")}
    if replaced is not None:
        out["replaced"] = replaced.model_dump(mode="json")
    return out


@mcp.tool
async def memory_get(ctx: Context, memory_id: str) -> dict[str, Any] | None:
    """Fetch a single memory by id (only if it's in a scope you can access)."""
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), None)
    svc = await _get_service()
    memory = await svc.get(memory_id, scopes)
    return memory.model_dump(mode="json") if memory else None


@mcp.tool
async def memory_update(
    ctx: Context,
    memory_id: str,
    content: str | None = None,
    description: str | None = None,
    type: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    importance: float | None = None,
) -> dict[str, Any] | None:
    """Edit an existing memory by id (only if it's in a scope you can access).

    Only the fields you pass are changed; the rest are preserved. Use this to fix
    or extend a known memory, retag it, or adjust its `importance` — instead of
    re-saving and relying on dedup. Returns the updated memory, or null if not found.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), None)
    svc = await _get_service()
    memory = await svc.update(
        memory_id,
        scopes,
        content=content,
        description=description,
        type=type,
        tags=tags,
        metadata=metadata,
        importance=importance,
    )
    return memory.model_dump(mode="json") if memory else None


@mcp.tool
async def memory_list(
    ctx: Context,
    scope: str | None = None,
    tags: list[str] | None = None,
    limit: int = 50,
    include_archived: bool = False,
    full: bool = False,
) -> list[dict[str, Any]]:
    """Browse recent memories (newest first). A cheap index of what's stored.

    Returns each memory's `description` (the one-line summary) and metadata but
    **not** its `content` — that is what makes this cheap enough to call for
    orientation: content runs ~11x longer than the description and would be ~74%
    of the payload. Scan the descriptions, then `memory_get(memory_id)` the one
    you actually need. Pass `full: true` only when you genuinely need every
    body at once (e.g. an audit), and prefer a smaller `limit` with it.

    Defaults to the current project plus shared scopes, and hides archived
    (forgotten) memories. Pass `include_archived: true` to review what was archived
    (an `archived_at` timestamp is set on those) — e.g. before restoring one.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), scope)
    svc = await _get_service()
    items = await svc.list(scopes, tags=tags, limit=limit, include_archived=include_archived)
    exclude = set() if full else {"content"}
    return [m.model_dump(mode="json", exclude=exclude) for m in items]


@mcp.tool
async def memory_restore(ctx: Context, memory_id: str) -> dict[str, Any] | None:
    """Un-archive a forgotten memory so it shows up in recall again.

    The inverse of `memory_forget`. Find archived ids via
    `memory_list(include_archived=true)`. Returns the restored memory, or null if
    not found in a scope you can access.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), None)
    svc = await _get_service()
    memory = await svc.restore(memory_id, scopes)
    return memory.model_dump(mode="json") if memory else None


@mcp.tool
async def memory_delete(ctx: Context, memory_id: str) -> dict[str, bool]:
    """Delete a memory by id (only if it's in a scope you can access)."""
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), None)
    svc = await _get_service()
    deleted = await svc.delete(memory_id, scopes)
    return {"deleted": deleted}


@mcp.tool
async def memory_stats(
    ctx: Context,
    scope: str | None = None,
    days: float = 30.0,
) -> dict[str, Any]:
    """Recall health check: store size per scope + search-quality metrics.

    Reports active/archived memory counts for the scopes you can access, and —
    from the search log — search volume, empty-result rate, average hits and top
    score, and latency percentiles over the last `days`, plus the most recent
    queries that returned nothing (recall misses worth investigating). Use it to
    judge whether recall is healthy and to tune `min_similarity` from evidence.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), scope)
    svc = await _get_service()
    return await svc.stats(scopes, days=days)


@mcp.tool
async def memory_forget(
    ctx: Context,
    scope: str | None = None,
    tags: list[str] | None = None,
    older_than_days: float | None = None,
    importance_floor: float | None = None,
    apply: bool = False,
    limit: int = 100,
) -> dict[str, Any]:
    """Archive stale, low-importance memories so old clutter stops diluting recall.

    A memory is eligible when it hasn't been recalled in `older_than_days` AND its
    `importance` is at or below `importance_floor` — recall and a higher importance
    both protect it. Archived memories drop out of search/get/list but the rows are
    kept (recoverable), not hard-deleted.

    Defaults to a **dry run**: it reports what would be archived. Pass `apply: true`
    to actually archive. Operates over the scopes you can access (optionally narrowed
    by `tags`). `matched` is the true total the criteria hit — exactly what apply
    would archive — while `memories` lists at most `limit` of them (stalest first;
    `truncated` is set when the list was cut).
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), scope)
    settings = get_settings()
    svc = await _get_service()
    return await svc.forget(
        scopes,
        tags=tags,
        older_than_days=(
            settings.forget_after_days if older_than_days is None else older_than_days
        ),
        importance_floor=(
            settings.forget_importance_floor
            if importance_floor is None
            else importance_floor
        ),
        apply=apply,
        limit=limit,
    )


@mcp.tool
async def memory_usage_guide(ctx: Context, section: str = "all") -> dict[str, Any]:
    """Generate up-to-date instructions for using this memory server.

    Returns guidance built from *this* server's live settings, registered tools,
    and your session — so it states how the server actually behaves rather than
    how some copy of a README once described it. Call it when setting the server
    up, after upgrading it, or when asked to add/refresh memory guidelines.

    Sections (`section`, default "all"):
      - `claude_md` — a ready-to-write CLAUDE.md section telling an agent when to
        call each tool, with this server's real defaults filled in. Write it into
        the project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for every project),
        replacing any previous hypermnesia section rather than appending a second.
      - `setup` — client registration snippets for this server's address.
      - `session` — what your connection resolved to: principal, project scope,
        readable scopes, whether workspace roots were detected.

    Always check the returned `warnings`: that is where a silently degraded
    setup — most importantly, project isolation not being active — shows up.
    """
    p = _principal()
    project_scope, roots_ok = await _project_scope_with_roots(ctx)
    try:
        return build_guide(
            settings=get_settings(),
            tools=tool_summaries(globals().values()),
            principal_id=p.id,
            project_scope=project_scope,
            read_scopes=_read_scopes(p, project_scope, None),
            roots_resolved=roots_ok,
            section=section,
        )
    except ValueError as e:
        raise ToolError(str(e)) from e


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.setLevel(settings.log_level.upper())
    mcp.run(transport="http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
