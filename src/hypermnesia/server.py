from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers

from .auth import AuthError, require_write_scope, resolve_principal, resolve_scopes
from .config import Principal, get_settings
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


def _scopes(principal: Principal, requested: str | None) -> list[str]:
    try:
        return resolve_scopes(principal, requested)
    except AuthError as e:
        raise ToolError(str(e)) from e


@mcp.tool
async def memory_search(
    query: str,
    scope: str | None = None,
    tags: list[str] | None = None,
    k: int = 8,
) -> list[dict[str, Any]]:
    """Semantically recall memories relevant to `query`.

    Search this before starting a task. Omit `scope` to search every scope you
    can access. Returns hits ordered by similarity (0-1).
    """
    p = _principal()
    svc = await _get_service()
    hits = await svc.search(query=query, scopes=_scopes(p, scope), tags=tags, k=k)
    return [h.model_dump(mode="json") for h in hits]


@mcp.tool
async def memory_save(
    content: str,
    description: str,
    scope: str,
    type: str = "fact",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    importance: float = 1.0,
) -> dict[str, Any]:
    """Store a memory. `description` is a one-line summary used for ranking/dedup.

    If a near-duplicate already exists in `scope`, it is updated instead of
    creating a new row. `type` is one of fact | preference | project | reference.
    """
    p = _principal()
    try:
        require_write_scope(p, scope)
    except AuthError as e:
        raise ToolError(str(e)) from e
    svc = await _get_service()
    memory, created = await svc.save(
        owner_id=p.id,
        scope=scope,
        content=content,
        description=description,
        type=type,
        tags=tags,
        metadata=metadata,
        importance=importance,
    )
    return {"created": created, "memory": memory.model_dump(mode="json")}


@mcp.tool
async def memory_get(memory_id: str) -> dict[str, Any] | None:
    """Fetch a single memory by id (only if it's in a scope you can access)."""
    p = _principal()
    svc = await _get_service()
    memory = await svc.get(memory_id, _scopes(p, None))
    return memory.model_dump(mode="json") if memory else None


@mcp.tool
async def memory_list(
    scope: str | None = None, tags: list[str] | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    """Browse recent memories (newest first). Use as a cheap index of what's stored."""
    p = _principal()
    svc = await _get_service()
    items = await svc.list(_scopes(p, scope), tags=tags, limit=limit)
    return [m.model_dump(mode="json") for m in items]


@mcp.tool
async def memory_delete(memory_id: str) -> dict[str, bool]:
    """Delete a memory by id (only if it's in a scope you can access)."""
    p = _principal()
    svc = await _get_service()
    deleted = await svc.delete(memory_id, _scopes(p, None))
    return {"deleted": deleted}


def main() -> None:
    settings = get_settings()
    mcp.run(transport="http", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
