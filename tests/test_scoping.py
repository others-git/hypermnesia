import pytest

from hypermnesia.auth import AuthError, effective_read_scopes, effective_write_scope
from hypermnesia.config import Principal
from hypermnesia.scoping import derive_project_scope, project_id_from_root


def test_project_id_is_readable_and_stable():
    pid = project_id_from_root("file:///mnt/d/REPOS/hypermnesia")
    assert pid.startswith("hypermnesia-")
    # stable across calls
    assert pid == project_id_from_root("file:///mnt/d/REPOS/hypermnesia")
    # trailing slash doesn't change identity
    assert pid == project_id_from_root("file:///mnt/d/REPOS/hypermnesia/")


def test_same_basename_different_path_distinct():
    a = project_id_from_root("file:///home/a/widgets")
    b = project_id_from_root("file:///home/b/widgets")
    assert a != b
    assert a.startswith("widgets-") and b.startswith("widgets-")


def test_derive_precedence():
    # header override wins
    assert derive_project_scope(
        ["file:///x/y"], "Team Key"
    ) == "project:team-key"
    # else first root
    assert derive_project_scope(["file:///x/y"], None).startswith("project:y-")
    # else default
    assert derive_project_scope([], None) == "default"
    assert derive_project_scope([], "  ") == "default"


def test_write_defaults_to_project_scope():
    p = Principal(id="dev", scopes=("shared",))
    # no explicit scope -> the session's project scope, even if not granted
    assert effective_write_scope(p, "project:foo-123", None) == "project:foo-123"
    # explicit shared is a granted scope
    assert effective_write_scope(p, "project:foo-123", "shared") == "shared"
    # explicit other-project is refused
    with pytest.raises(AuthError):
        effective_write_scope(p, "project:foo-123", "project:bar-999")


def test_read_defaults_to_project_plus_grants():
    p = Principal(id="dev", scopes=("shared", "default"))
    scopes = effective_read_scopes(p, "project:foo-123", None)
    assert scopes[0] == "project:foo-123"
    assert set(scopes) == {"project:foo-123", "shared", "default"}
    # never leaks another project: requesting one you don't own is refused
    with pytest.raises(AuthError):
        effective_read_scopes(p, "project:foo-123", "project:bar-999")
    # requesting your own project scope explicitly is fine
    assert effective_read_scopes(p, "project:foo-123", "project:foo-123") == [
        "project:foo-123"
    ]
