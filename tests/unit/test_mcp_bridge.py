"""Tests for the MCP bridge and its registry integration.

Everything here runs offline against ``tests/fixtures/fake_github_mcp_server.py``,
a tiny FastMCP stdio server that imitates the GitHub MCP server's publish tools.
No Docker, no network, no real GitHub.

Covers:
  * MCPServerConnection: starts, discovers tools, calls them, normalizes
    results (text + structured), surfaces tool errors, stops cleanly.
  * mcp_integration.ensure_server: connects from config, registers prefixed
    tools, is idempotent, and degrades (raises) when the server is unavailable.
  * registry dispatch routes through the bridge end-to-end.
  * repo_publisher declares the github MCP server and its github_* tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from autoscientist.agents import get_agent
from autoscientist.clients.mcp_bridge import (
    HttpTransport,
    MCPBridgeError,
    MCPServerConnection,
    StdioTransport,
)
from autoscientist.runtime.config import Config, load_config
from autoscientist.tools import mcp_integration
from autoscientist.tools import registry as tool_registry

_REPO = Path(__file__).resolve().parents[2]
_FAKE_SERVER = _REPO / "tests" / "fixtures" / "fake_github_mcp_server.py"


def _fake_transport() -> StdioTransport:
    # Use the same interpreter running the tests so the SDK is importable.
    return StdioTransport(command=sys.executable, args=(str(_FAKE_SERVER),))


def _fake_config() -> Config:
    return Config(
        default={},
        models={},
        mcp={
            "servers": {
                "github": {
                    "enabled": True,
                    "transport": "stdio",
                    "tool_prefix": "github_",
                    "allowed_tools": [
                        "get_me",
                        "create_repository",
                        "push_files",
                        "create_or_update_file",
                    ],
                    "startup_timeout_s": 30,
                    "stdio": {
                        "command": sys.executable,
                        "args": [str(_FAKE_SERVER)],
                    },
                }
            }
        },
        root=_REPO,
    )


# ---------------------------------------------------------------------------
# MCPServerConnection (transport layer)
# ---------------------------------------------------------------------------

def test_connection_starts_and_lists_tools() -> None:
    with MCPServerConnection(_fake_transport(), name="fake", startup_timeout_s=30) as conn:
        names = {t.name for t in conn.list_tools()}
        assert {"get_me", "create_repository", "push_files"} <= names
        # Schemas are real JSON Schema objects.
        push = next(t for t in conn.list_tools() if t.name == "push_files")
        assert push.input_schema["type"] == "object"
        assert "files" in push.input_schema["properties"]


def test_connection_call_tool_returns_structured() -> None:
    with MCPServerConnection(_fake_transport(), name="fake", startup_timeout_s=30) as conn:
        out = conn.call_tool("get_me", {})
        assert out["is_error"] is False
        assert out["structured"]["login"] == "test-operator"


def test_connection_publish_flow() -> None:
    with MCPServerConnection(_fake_transport(), name="fake", startup_timeout_s=30) as conn:
        repo = conn.call_tool(
            "create_repository",
            {"name": "demo", "autoInit": True, "private": True},
        )
        assert repo["structured"]["full_name"] == "test-operator/demo"
        pushed = conn.call_tool(
            "push_files",
            {
                "owner": "test-operator",
                "repo": "demo",
                "branch": "main",
                "message": "init",
                "files": [
                    {"path": "README.md", "content": "# demo"},
                    {"path": "src/a.py", "content": "x = 1\n"},
                ],
            },
        )
        assert pushed["is_error"] is False
        assert pushed["structured"]["files_pushed"] == 2


def test_connection_push_before_branch_exists_is_tool_error() -> None:
    # autoInit defaults to False -> no 'main' branch -> push must report error.
    with MCPServerConnection(_fake_transport(), name="fake", startup_timeout_s=30) as conn:
        conn.call_tool("create_repository", {"name": "norepo"})
        out = conn.call_tool(
            "push_files",
            {
                "owner": "test-operator",
                "repo": "norepo",
                "branch": "main",
                "message": "x",
                "files": [{"path": "f.txt", "content": "y"}],
            },
        )
        assert out["is_error"] is True


def test_call_before_start_raises() -> None:
    conn = MCPServerConnection(_fake_transport(), name="fake")
    with pytest.raises(MCPBridgeError):
        conn.call_tool("get_me", {})


def test_auth_secrets_not_in_repr() -> None:
    # A leaked PAT in a repr ends up in logs/tracebacks; the secret-bearing
    # fields must be repr=False.
    secret = "ghp_SUPERSECRET_TOKEN_do_not_leak"
    st = StdioTransport(command="x", env={"GITHUB_PERSONAL_ACCESS_TOKEN": secret})
    ht = HttpTransport(url="https://e/mcp/", headers={"Authorization": f"Bearer {secret}"})
    assert secret not in repr(st)
    assert secret not in repr(ht)
    assert secret not in repr(MCPServerConnection(st, name="redact"))


def test_start_failure_raises_cleanly() -> None:
    # A command that exits immediately can never complete the MCP handshake.
    bad = StdioTransport(command=sys.executable, args=("-c", "raise SystemExit(1)"))
    conn = MCPServerConnection(bad, name="bad", startup_timeout_s=10)
    with pytest.raises(MCPBridgeError):
        conn.start()
    conn.stop()  # idempotent / safe even though start failed


# ---------------------------------------------------------------------------
# mcp_integration (registry wiring)
# ---------------------------------------------------------------------------

def test_ensure_server_registers_prefixed_tools() -> None:
    cfg = _fake_config()
    try:
        names = mcp_integration.ensure_server("github", cfg)
        assert "github_create_repository" in names
        assert "github_push_files" in names
        assert tool_registry.is_registered("github_get_me")
        # Idempotent: second call returns the same set, no duplicate error.
        assert mcp_integration.ensure_server("github", cfg) == names
    finally:
        mcp_integration.shutdown_all()


def test_ensure_server_dispatch_round_trip() -> None:
    cfg = _fake_config()
    try:
        mcp_integration.ensure_server("github", cfg)
        ctx = tool_registry.ToolContext()
        dr = tool_registry.dispatch(
            "github_create_repository",
            {"name": "demo", "autoInit": True},
            ctx,
        )
        assert dr.error is None
        assert dr.output["structured"]["full_name"] == "test-operator/demo"
    finally:
        mcp_integration.shutdown_all()


def test_ensure_server_allowlist_filters() -> None:
    cfg = _fake_config()
    # create_branch exists on the fake server but is NOT in the allowlist.
    try:
        names = mcp_integration.ensure_server("github", cfg)
        assert "github_create_branch" not in names
        assert not tool_registry.is_registered("github_create_branch")
    finally:
        mcp_integration.shutdown_all()


def test_ensure_server_unknown_key_raises() -> None:
    cfg = _fake_config()
    with pytest.raises(mcp_integration.MCPConfigError):
        mcp_integration.ensure_server("nope", cfg)


def test_ensure_server_missing_token_raises() -> None:
    # An http server whose auth needs a token that isn't set must raise, so the
    # runner can degrade gracefully rather than connect unauthenticated.
    cfg = Config(
        default={},
        models={},
        mcp={
            "servers": {
                "remote": {
                    "transport": "http",
                    "http": {
                        "url": "https://example.invalid/mcp/",
                        "token_env": "DEFINITELY_UNSET_TOKEN_VAR_XYZ",
                        "auth_value": "Bearer {token}",
                    },
                }
            }
        },
        root=_REPO,
    )
    with pytest.raises(mcp_integration.MCPConfigError):
        mcp_integration.ensure_server("remote", cfg)


# ---------------------------------------------------------------------------
# Agent wiring
# ---------------------------------------------------------------------------

def test_repo_publisher_declares_github_mcp() -> None:
    cfg = load_config()
    agent = get_agent("repo_publisher", cfg)
    assert agent is not None
    assert "github" in agent.mcp_servers
    assert "github_create_repository" in agent.tools
    assert "github_push_files" in agent.tools
    # Native tools are still present.
    assert "write_release_file" in agent.tools
