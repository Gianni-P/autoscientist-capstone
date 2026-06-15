"""Pre-flight for the GitHub MCP integration: one real, READ-ONLY round-trip.

Validates live wiring to the GitHub MCP server configured in config/mcp.toml
(remote Streamable HTTP by default; local Docker if you switched transport)
before repo_publisher tries to publish for real. It:

  1. confirms GITHUB_PERSONAL_ACCESS_TOKEN is set,
  2. connects to the server with the exact transport the runner would build,
  3. confirms the publish tools (create_repository, push_files,
     create_or_update_file) are exposed,
  4. makes one READ-ONLY call — search_repositories for "user:@me" — which both
     proves the token authenticates against the GitHub API and reveals which
     account it belongs to.

It makes NO writes: no repo is created, nothing is pushed. Cost: $0 (no LLM).
(The remote GitHub MCP server exposes only the 'repos' toolset and has no
get_me tool, so repo_publisher reads the owner from create_repository's
response; this pre-flight uses search_repositories to confirm identity.)

    uv run python scripts/dry_run_github_mcp.py

If this passes, a live repo_publisher run can publish; if the token or network
is missing it fails here cleanly instead of mid-run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Publish tools repo_publisher needs the server to expose.
_REQUIRED_TOOLS = ("create_repository", "push_files", "create_or_update_file")


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(failures: list[str], msg: str) -> None:
    print(f"  FAIL  {msg}")
    failures.append(msg)


def main() -> int:
    from autoscientist.clients.mcp_bridge import MCPServerConnection
    from autoscientist.runtime.config import load_config
    from autoscientist.tools.mcp_integration import _build_transport, _server_config

    failures: list[str] = []
    cfg = load_config(reload=True)

    # ------------------------------------------------------------------
    section("Config: github MCP server present in config/mcp.toml")
    servers = (cfg.mcp or {}).get("servers", {})
    if "github" not in servers:
        fail(failures, "no [servers.github] in config/mcp.toml")
        return _finish(failures)
    sc = servers["github"]
    transport_kind = sc.get("transport", "http")
    passed(f"github server configured (transport={transport_kind})")

    # ------------------------------------------------------------------
    section("Credentials: GITHUB_PERSONAL_ACCESS_TOKEN present")
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        fail(failures, "GITHUB_PERSONAL_ACCESS_TOKEN is unset (export it or add to .env)")
        return _finish(failures)
    passed(f"token present (len={len(token)})")

    # ------------------------------------------------------------------
    section("Connect: reach the GitHub MCP server (same transport as a real run)")
    try:
        transport = _build_transport("github", _server_config(cfg, "github"))
        conn = MCPServerConnection(
            transport, name="github",
            startup_timeout_s=float(sc.get("startup_timeout_s", 60.0)),
        )
        conn.start()
    except Exception as e:
        fail(failures, f"could not connect: {type(e).__name__}: {e}")
        if transport_kind == "stdio":
            print("        (stdio transport needs Docker reachable without sudo — "
                  "see config/mcp.toml, or switch transport=\"http\")")
        return _finish(failures)

    try:
        tool_names = {td.name for td in conn.list_tools()}
        passed(f"connected; server exposes {len(tool_names)} tools")
        for n in _REQUIRED_TOOLS:
            if n in tool_names:
                passed(f"  publish tool available: {n}")
            else:
                fail(failures, f"  required publish tool missing: {n}")

        # ------------------------------------------------------------------
        section("Read-only call: search_repositories for user:@me (no writes)")
        out = conn.call_tool("search_repositories", {"query": "user:@me", "perPage": 1})
        if out["is_error"]:
            fail(failures, f"search_repositories errored (token rejected?): {out.get('text')}")
        else:
            structured = out.get("structured") or {}
            items = structured.get("items") if isinstance(structured, dict) else None
            owner = None
            if items:
                first = items[0]
                owner = (first.get("owner") or {}).get("login") or (
                    (first.get("full_name") or "/").split("/")[0]
                )
            if owner:
                passed(f"authenticated against GitHub as: {owner}")
            else:
                # Auth worked (no error) but the account has no repos to name.
                passed("token authenticated (account has no repositories to name)")
    finally:
        conn.stop()

    return _finish(failures)


def _finish(failures: list[str]) -> int:
    print()
    if failures:
        print(f"*** GitHub MCP dry run FAILED: {len(failures)} issue(s) ***")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("*** GitHub MCP dry run PASSED — live wiring is good, repo_publisher can publish. ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())
