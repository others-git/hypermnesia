from __future__ import annotations

from .config import Principal, Settings


class AuthError(Exception):
    """Raised when a caller cannot be authenticated or authorized."""


def bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None
    parts = authorization_header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def resolve_principal(settings: Settings, authorization_header: str | None) -> Principal:
    if not settings.require_auth:
        return Principal(id="anonymous", scopes=("default",))
    token = bearer_token(authorization_header)
    if not token:
        raise AuthError("Missing or malformed Authorization: Bearer <token> header.")
    principal = settings.principals().get(token)
    if principal is None:
        raise AuthError("Invalid bearer token.")
    return principal


def resolve_scopes(principal: Principal, requested: str | None) -> list[str]:
    """Read scopes: a specific requested scope (must be allowed) or all allowed."""
    if requested is None:
        return list(principal.scopes)
    if not principal.may_access(requested):
        raise AuthError(f"Principal {principal.id!r} may not access scope {requested!r}.")
    return [requested]


def require_write_scope(principal: Principal, scope: str) -> None:
    if not principal.may_access(scope):
        raise AuthError(f"Principal {principal.id!r} may not write to scope {scope!r}.")
