from __future__ import annotations

import hashlib
import re
from urllib.parse import unquote, urlparse

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.lower()).strip("-") or "x"


def project_id_from_root(root_uri: str) -> str:
    """Derive a stable, human-readable project id from a workspace root URI.

    The basename keeps it readable; an 8-char hash of the full path keeps it
    unique across same-named directories.

        file:///mnt/d/REPOS/hypermnesia -> hypermnesia-3f9a2c11
    """
    parsed = urlparse(root_uri)
    path = (unquote(parsed.path) or root_uri).rstrip("/")
    base = path.rsplit("/", 1)[-1] if "/" in path else path
    digest = hashlib.sha256(path.encode()).hexdigest()[:8]
    return f"{_slug(base)}-{digest}"


def derive_project_scope(roots: list[str], header_override: str | None) -> str:
    """The scope a session's memories default to.

    Precedence:
      1. ``X-Hypermnesia-Project`` header  -> ``project:<slug>``  (stable/team key)
      2. first workspace root              -> ``project:<id-from-path>`` (automatic)
      3. neither                           -> ``default``
    """
    if header_override and header_override.strip():
        return f"project:{_slug(header_override)}"
    if roots:
        return f"project:{project_id_from_root(roots[0])}"
    return "default"
