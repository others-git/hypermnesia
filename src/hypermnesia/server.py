from __future__ import annotations

import asyncio
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
from .scoping import derive_project_scope
from .service import MemoryService

mcp = FastMCP("hypermnesia")

_service: MemoryService | None = None
_lock = asyncio.Lock()


async def _get_service() -> MemoryService:
    global _service
    if _service is None:
        async with _lock:
            if _service is None:
                _service = await MemoryService.create(get_settings())
    return _service


def _principal() -> Principal:
    # include_all=True so the Authorization header isn't filtered out.
    headers = get_http_headers(include_all=True)
    try:
        return resolve_principal(get_settings(), headers.get("authorization"))
    except AuthError as e:
        raise ToolError(str(e)) from e


async def _project_scope(ctx: Context) -> str:
    """Derive the calling session's project scope from its workspace root.

    Clients (e.g. Claude Code) advertise the project directory as an MCP root;
    a ``X-Hypermnesia-Project`` header overrides it for stable/shared keys.
    """
    headers = get_http_headers(include_all=True)
    override = headers.get("x-hypermnesia-project")
    roots: list[str] = []
    try:
        result = await ctx.list_roots()
        raw = getattr(result, "roots", result)
        roots = [str(getattr(r, "uri", r)) for r in raw]
    except Exception:  # noqa: BLE001 - clients without roots support fall back
        roots = []
    return derive_project_scope(roots, override)


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
) -> list[dict[str, Any]]:
    """Semantically recall memories relevant to `query`.

    Search this before starting a task. By default searches the current project's
    memories plus any shared scopes — never another project's. Results are ranked
    by a blend of semantic `similarity` (0-1), recency, and importance, exposed as
    `score`. Pass `min_similarity` (e.g. 0.3) to drop weak matches.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), scope)
    svc = await _get_service()
    hits = await svc.search(
        query=query, scopes=scopes, tags=tags, k=k, min_similarity=min_similarity
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
    ctx: Context, scope: str | None = None, tags: list[str] | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Browse recent memories (newest first). Use as a cheap index of what's stored.

    Defaults to the current project plus shared scopes.
    """
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), scope)
    svc = await _get_service()
    items = await svc.list(scopes, tags=tags, limit=limit)
    return [m.model_dump(mode="json") for m in items]


@mcp.tool
async def memory_delete(ctx: Context, memory_id: str) -> dict[str, bool]:
    """Delete a memory by id (only if it's in a scope you can access)."""
    p = _principal()
    scopes = _read_scopes(p, await _project_scope(ctx), None)
    svc = await _get_service()
    deleted = await svc.delete(memory_id, scopes)
    return {"deleted": deleted}


@mcp.tool
async def memory_forget(
    ctx: Context,
    scope: str | None = None,
    tags: list[str] | None = None,
    older_than_days: float | None = None,
    importance_floor: float | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Archive stale, low-importance memories so old clutter stops diluting recall.

    A memory is eligible when it hasn't been recalled in `older_than_days` AND its
    `importance` is at or below `importance_floor` — recall and a higher importance
    both protect it. Archived memories drop out of search/get/list but the rows are
    kept (recoverable), not hard-deleted.

    Defaults to a **dry run**: it reports what would be archived. Pass `apply: true`
    to actually archive. Operates over the scopes you can access (optionally narrowed
    by `tags`).
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
    )


def main() -> None:
    settings = get_settings()
    mcp.run(transport="http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
