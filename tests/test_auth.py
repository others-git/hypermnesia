import pytest

from hypermnesia.auth import (
    AuthError,
    bearer_token,
    resolve_principal,
    resolve_scopes,
)
from hypermnesia.config import Principal, Settings


def test_bearer_token_parsing():
    assert bearer_token("Bearer abc123") == "abc123"
    assert bearer_token("bearer abc123") == "abc123"
    assert bearer_token("Basic abc123") is None
    assert bearer_token(None) is None
    assert bearer_token("") is None


def _settings(**kw) -> Settings:
    base = dict(
        require_auth=True,
        auth_tokens='{"tok":{"principal":"agent-a","scopes":["shared","user:dev-test"]}}',
    )
    base.update(kw)
    return Settings(**base)


def test_resolve_principal_valid():
    p = resolve_principal(_settings(), "Bearer tok")
    assert p.id == "agent-a"
    assert "shared" in p.scopes


def test_resolve_principal_rejects_bad_token():
    with pytest.raises(AuthError):
        resolve_principal(_settings(), "Bearer nope")
    with pytest.raises(AuthError):
        resolve_principal(_settings(), None)


def test_resolve_principal_no_auth_mode():
    p = resolve_principal(_settings(require_auth=False), None)
    assert p.id == "anonymous"
    assert p.scopes == ("default",)


def test_resolve_scopes():
    p = Principal(id="x", scopes=("shared", "user:dev-test"))
    assert set(resolve_scopes(p, None)) == {"shared", "user:dev-test"}
    assert resolve_scopes(p, "shared") == ["shared"]
    with pytest.raises(AuthError):
        resolve_scopes(p, "user:bob")
