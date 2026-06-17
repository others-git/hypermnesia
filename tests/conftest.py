from __future__ import annotations

import os

import pytest

E2E_URL = os.environ.get("HM_TEST_URL", "http://localhost:8000/mcp")
E2E_TOKEN = os.environ.get("HM_TEST_TOKEN", "dev-token")
E2E_SCOPE = os.environ.get("HM_TEST_SCOPE", "user:dev-test")


def make_client(
    token: str | None = E2E_TOKEN,
    roots: list[str] | None = None,
    project_header: str | None = None,
):
    """Construct an MCP client for the running server (lazy imports for unit-only runs).

    `roots` advertises workspace roots (like an editor/agent), driving automatic
    project scoping. `project_header` sets the X-Hypermnesia-Project override.
    """
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if project_header:
        headers["X-Hypermnesia-Project"] = project_header
    transport = StreamableHttpTransport(E2E_URL, headers=headers)
    return Client(transport, roots=roots) if roots else Client(transport)
