"""Unit tests for the workspace-roots shim behind project scoping.

FastMCP 4 removed ``Context.list_roots`` while keeping the session method, and
the failure was silent: every project fell back to the ``default`` scope and
shared one memory pool. These pin the shim to both shapes.
"""

from __future__ import annotations

import pytest

from hypermnesia.server import _list_roots


class _Result:
    """A response object that wraps the list, as older FastMCP returns."""

    def __init__(self, roots):
        self.roots = roots


class _Root:
    def __init__(self, uri):
        self.uri = uri


class _Session:
    def __init__(self, roots=None, exc=None):
        self._roots, self._exc = roots, exc

    async def list_roots(self):
        if self._exc:
            raise self._exc
        return self._roots


class _Ctx:
    """Pre-4 shape: Context.list_roots exists."""

    def __init__(self, *, ctx_roots=None, ctx_exc=None, session=None):
        self.session = session
        self._ctx_roots, self._ctx_exc = ctx_roots, ctx_exc

    async def list_roots(self):
        if self._ctx_exc:
            raise self._ctx_exc
        return self._ctx_roots


class _Ctx4:
    """FastMCP 4 shape: no Context.list_roots at all, session method present."""

    def __init__(self, session):
        self.session = session


async def test_uses_context_list_roots_when_present():
    ctx = _Ctx(ctx_roots=_Result([_Root("file:///w/alpha")]), session=None)
    assert await _list_roots(ctx) == ["file:///w/alpha"]


async def test_falls_back_to_session_when_context_api_is_gone():
    # The FastMCP 4 regression: without this fallback scoping collapses.
    ctx = _Ctx4(_Session(_Result([_Root("file:///w/beta")])))
    assert await _list_roots(ctx) == ["file:///w/beta"]


async def test_falls_back_to_session_when_context_api_raises():
    ctx = _Ctx(
        ctx_exc=AttributeError("no attribute 'list_roots'"),
        session=_Session(_Result([_Root("file:///w/gamma")])),
    )
    assert await _list_roots(ctx) == ["file:///w/gamma"]


async def test_accepts_a_bare_list_response():
    # Some versions return the list directly rather than a wrapper object.
    ctx = _Ctx4(_Session([_Root("file:///w/delta")]))
    assert await _list_roots(ctx) == ["file:///w/delta"]


async def test_no_roots_support_anywhere_returns_empty():
    # A client with no roots capability is supported: fall back, don't raise.
    ctx = _Ctx4(_Session(exc=RuntimeError("client does not support roots")))
    assert await _list_roots(ctx) == []


async def test_empty_roots_list_is_returned_as_empty():
    ctx = _Ctx4(_Session(_Result([])))
    assert await _list_roots(ctx) == []


@pytest.mark.parametrize("roots", [[], [_Root("file:///w/x")]])
async def test_never_raises_regardless_of_shape(roots):
    assert isinstance(await _list_roots(_Ctx4(_Session(_Result(roots)))), list)
