"""Offline smoke test for the GitHub MCP integration.

Exercises the whole substrate WITHOUT Docker, network, or a real GitHub:
a local FastMCP stdio server (tests/fixtures/fake_github_mcp_server.py) stands
in for the official GitHub MCP server, with the same publish tool names/shapes.

What it proves:
  1. mcp_integration.ensure_server connects from config and registers the
     server's tools into the registry under the github_ prefix.
  2. The registry dispatches a create_repository + push_files round-trip
     through the bridge to the (fake) server.
  3. The runner's tool-resolution path exposes the agent's github_* tools when
     the server is up...
  4. ...and gracefully DROPS them (no crash) when the server is unavailable —
     the same path that keeps a live run alive when GITHUB_PERSONAL_ACCESS_TOKEN
     is missing or Docker is down.

    uv run python scripts/smoke_mcp_github.py

Cost: $0 (no LLM calls).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_FAKE_SERVER = _REPO / "tests" / "fixtures" / "fake_github_mcp_server.py"


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(failures: list[str], msg: str) -> None:
    print(f"  FAIL  {msg}")
    failures.append(msg)


def _fake_config(command: str, args: list[str]):
    from autoscientist.runtime.config import Config

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
                    "stdio": {"command": command, "args": args},
                }
            }
        },
        root=_REPO,
    )


def main() -> int:
    from autoscientist.runtime.agent import Agent
    from autoscientist.tools import mcp_integration
    from autoscientist.tools import registry as tool_registry

    failures: list[str] = []

    # ------------------------------------------------------------------
    section("Connect + register: fake GitHub MCP server over stdio")
    cfg = _fake_config(sys.executable, [str(_FAKE_SERVER)])
    try:
        names = mcp_integration.ensure_server("github", cfg)
    except Exception as e:
        fail(failures, f"ensure_server raised {type(e).__name__}: {e}")
        return _finish(failures)
    for expected in ("github_get_me", "github_create_repository", "github_push_files"):
        if expected in names:
            passed(f"registered {expected}")
        else:
            fail(failures, f"{expected} not registered (got {names})")
    if mcp_integration.ensure_server("github", cfg) == names:
        passed("ensure_server is idempotent")
    else:
        fail(failures, "ensure_server not idempotent")

    # ------------------------------------------------------------------
    section("Dispatch: create_repository + push_files through the registry")
    ctx = tool_registry.ToolContext()
    dr = tool_registry.dispatch(
        "github_create_repository",
        {"name": "pneumonia-data-efficiency", "autoInit": True, "private": True},
        ctx,
    )
    if dr.error is None and dr.output["structured"]["full_name"].endswith("pneumonia-data-efficiency"):
        passed(f"create_repository -> {dr.output['structured']['html_url']}")
    else:
        fail(failures, f"create_repository failed: error={dr.error} output={dr.output}")

    dr2 = tool_registry.dispatch(
        "github_push_files",
        {
            "owner": "test-operator",
            "repo": "pneumonia-data-efficiency",
            "branch": "main",
            "message": "Publish reproducible release",
            "files": [
                {"path": "README.md", "content": "# Pneumonia data efficiency\n"},
                {"path": "src/datasets.py", "content": "# data\n"},
                {"path": "reproduce.sh", "content": "#!/usr/bin/env bash\n"},
            ],
        },
        ctx,
    )
    if dr2.error is None and dr2.output["structured"]["files_pushed"] == 3:
        passed("push_files committed 3 files in one commit")
    else:
        fail(failures, f"push_files failed: error={dr2.error} output={dr2.output}")

    # ------------------------------------------------------------------
    section("Runner path: agent github_* tools resolve when server is up")
    agent = Agent(
        name="repo_publisher",
        role="x",
        system_prompt_path=Path("repo_publisher.md"),
        tools=(
            "list_sandbox",
            "github_get_me",
            "github_create_repository",
            "github_push_files",
        ),
        mcp_servers=("github",),
    )
    effective = [t for t in agent.tools if tool_registry.is_registered(t)]
    if "github_push_files" in effective and "list_sandbox" in effective:
        passed(f"effective tools include github_* + native ({len(effective)} total)")
    else:
        fail(failures, f"expected github_* in effective tools, got {effective}")

    mcp_integration.shutdown_all()

    # ------------------------------------------------------------------
    section("Graceful degradation: server unavailable -> github_* dropped, no crash")
    # shutdown_all() above retracted the github_* tools; a failed (re)connect is
    # exactly what a live run sees with no token / Docker down.
    if not tool_registry.is_registered("github_push_files"):
        passed("shutdown_all retracted the github_* tools")
    else:
        fail(failures, "github_push_files still registered after shutdown_all")

    bad_cfg = _fake_config(sys.executable, ["-c", "import sys; sys.exit(1)"])
    degraded = False
    try:
        mcp_integration.ensure_server("github", bad_cfg)
    except Exception as e:
        degraded = True
        passed(f"ensure_server raised as expected: {type(e).__name__}")
    if not degraded:
        fail(failures, "ensure_server should have failed against a bad command")

    # The runner catches that exception and proceeds with native tools only:
    # the real github_* tools the agent declares are simply not registered now.
    native_only_agent = Agent(
        name="repo_publisher",
        role="x",
        system_prompt_path=Path("repo_publisher.md"),
        tools=("list_sandbox", "github_push_files", "github_create_repository"),
        mcp_servers=("github",),
    )
    effective2 = [t for t in native_only_agent.tools if tool_registry.is_registered(t)]
    dropped = [t for t in native_only_agent.tools if t not in effective2]
    if effective2 == ["list_sandbox"] and "github_push_files" in dropped:
        passed(f"unavailable github_* dropped ({dropped}); native tool retained")
    else:
        fail(failures, f"degradation filter wrong: effective={effective2} dropped={dropped}")

    mcp_integration.shutdown_all()
    return _finish(failures)


def _finish(failures: list[str]) -> int:
    print()
    if failures:
        print(f"*** smoke_mcp_github FAILED: {len(failures)} issue(s) ***")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("*** smoke_mcp_github PASSED — MCP/GitHub substrate is wired correctly. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
