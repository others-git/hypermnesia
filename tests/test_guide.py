"""Unit tests for the generated usage guide (pure functions, no server).

The guide exists because hand-maintained copies of this guidance drift out of
sync with the code. So the tests that matter are the ones proving it is really
*derived* — from live settings and the registered tool list — rather than a
second hardcoded copy that can rot the same way.
"""

from __future__ import annotations

import pytest

from hypermnesia.config import Settings
from hypermnesia.guide import (
    build,
    claude_md_block,
    session_report,
    setup_commands,
    tool_summaries,
    warnings_for,
)

TOOLS = [
    ("memory_search", "Semantically recall memories relevant to `query`."),
    ("memory_save", "Store a memory."),
    ("memory_list", "Browse recent memories."),
]


# --- tool discovery ---------------------------------------------------------


class _Fn:
    """FastMCP 3 shape: the decorator returns the plain function."""

    def __init__(self, name, doc):
        self.__name__, self.__doc__ = name, doc


class _Wrapped:
    """FastMCP 4 shape: a wrapper object carrying name/description."""

    def __init__(self, name, desc):
        self.name, self.description = name, desc


def test_reads_plain_functions_and_wrapper_objects():
    objs = [
        _Fn("memory_search", "First line.\n\nSecond paragraph."),
        _Wrapped("memory_save", "Store a memory."),
    ]
    assert tool_summaries(objs) == [
        ("memory_save", "Store a memory."),
        ("memory_search", "First line."),
    ]


def test_ignores_non_tool_module_globals():
    # A module's namespace is full of imports, constants and classes; some have
    # a `.name` that is not even a string.
    class HasNonStringName:
        name = property(lambda self: None)

    objs = [_Fn("helper", "not a tool"), HasNonStringName(), 42, "a string", None]
    assert tool_summaries(objs) == []


def test_a_new_tool_appears_without_touching_the_guide():
    # The anti-drift property: adding a tool must not require editing guide.py.
    tools = TOOLS + [("memory_teleport", "Does something new.")]
    block = claude_md_block(Settings(), tools)
    assert "memory_teleport" in block


# --- generated from live settings, not hardcoded ----------------------------


@pytest.mark.parametrize("cutoff", [0.15, 0.3])
def test_claude_md_states_the_configured_cutoff(cutoff):
    block = claude_md_block(Settings(search_relative_cutoff=cutoff), TOOLS)
    assert f"within {cutoff} similarity" in block


def test_claude_md_describes_a_disabled_cutoff_differently():
    block = claude_md_block(
        Settings(search_relative_cutoff=0.0, search_min_similarity=0.42), TOOLS
    )
    assert "disabled on this server" in block
    assert "0.42 similarity floor" in block


def test_claude_md_reflects_forget_and_dedupe_settings():
    block = claude_md_block(
        Settings(
            forget_after_days=30,
            forget_importance_floor=0.5,
            dedupe_threshold=0.8,
            dedupe_content_threshold=0.7,
        ),
        TOOLS,
    )
    assert "30 days" in block and "at or below 0.5" in block
    assert ">= 0.8" in block and ">= 0.7" in block


def test_claude_md_names_the_sessions_own_project_scope():
    block = claude_md_block(Settings(), TOOLS, project_scope="project:alpha-123")
    assert "project:alpha-123" in block


def test_claude_md_wraps_prose_to_a_readable_width():
    block = claude_md_block(Settings(), TOOLS, project_scope="project:" + "x" * 60)
    assert max(len(ln) for ln in block.splitlines()) <= 79


# --- setup snippets ---------------------------------------------------------


def test_setup_uses_the_servers_real_address():
    out = setup_commands(Settings(host="10.0.0.5", port=9000))
    assert "http://10.0.0.5:9000/mcp" in out


def test_setup_rewrites_a_bind_all_host_to_something_dialable():
    # 0.0.0.0 is a bind address, not somewhere a client can connect to.
    out = setup_commands(Settings(host="0.0.0.0"))
    assert "http://localhost:8765/mcp" in out
    # ...and says so, because the server cannot know the client's route to it.
    assert "is a guess" in out


def test_setup_does_not_hedge_a_real_routable_host():
    assert "is a guess" not in setup_commands(Settings(host="10.0.0.5"))


def test_setup_never_prints_real_tokens():
    secret = "s3cret-token-value"
    out = setup_commands(
        Settings(auth_tokens='{"%s": {"principal": "p", "scopes": ["a"]}}' % secret)
    )
    assert secret not in out
    assert "<your-token>" in out


# --- session diagnosis ------------------------------------------------------


def test_session_report_flags_inactive_isolation():
    r = session_report("dev", "default", ["default"], roots_resolved=False)
    assert r["project_isolation"].startswith("NOT ACTIVE")


def test_session_report_reports_active_isolation():
    r = session_report("dev", "project:a", ["project:a", "shared"], roots_resolved=True)
    assert r["project_isolation"] == "active"
    assert r["saves_default_to"] == "project:a"


def test_warns_when_every_project_shares_the_default_scope():
    (w,) = warnings_for("default", False, Settings())
    assert "NOT isolated" in w and "X-Hypermnesia-Project" in w


def test_header_derived_scope_is_reported_as_fine():
    (w,) = warnings_for("project:a", False, Settings())
    assert "no action needed" in w


def test_no_warning_when_roots_worked():
    assert warnings_for("project:a", True, Settings()) == []


def test_warns_when_auth_is_disabled():
    ws = warnings_for("project:a", True, Settings(require_auth=False))
    assert any("HM_REQUIRE_AUTH" in w for w in ws)


# --- assembly ---------------------------------------------------------------


def _build(section="all", **kw):
    args = dict(
        settings=Settings(),
        tools=TOOLS,
        principal_id="dev",
        project_scope="project:a",
        read_scopes=["project:a", "shared"],
        roots_resolved=True,
    )
    args.update(kw)
    return build(section=section, **args)


def test_all_sections_by_default():
    g = _build()
    assert set(g) == {"claude_md", "setup", "session", "warnings"}


@pytest.mark.parametrize("section", ["claude_md", "setup", "session"])
def test_a_single_section_can_be_requested(section):
    g = _build(section)
    assert set(g) == {section, "warnings"}  # warnings always ride along


def test_warnings_are_always_present_even_for_one_section():
    g = _build("setup", project_scope="default")
    assert g["warnings"] and "NOT isolated" in g["warnings"][0]


def test_setup_key_comes_from_the_real_project_scope():
    assert "hypermnesia-b7a8c6ec" in _build(project_scope="project:hypermnesia-b7a8c6ec")["setup"]


def test_setup_key_falls_back_to_a_placeholder_off_project():
    assert "my-project" in _build(project_scope="default")["setup"]


def test_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="unknown section"):
        _build("nonsense")
