from __future__ import annotations

import os

import pytest

E2E_URL = os.environ.get("HM_TEST_URL", "http://localhost:8000/mcp")
E2E_TOKEN = os.environ.get("HM_TEST_TOKEN", "dev-token")
E2E_SCOPE = os.environ.get("HM_TEST_SCOPE", "user:dev-test")


def make_client(token: str | None = E2E_TOKEN):
    """Construct an MCP client for the running server (lazy imports for unit-only runs)."""
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return Client(StreamableHttpTransport(E2E_URL, headers=headers))
