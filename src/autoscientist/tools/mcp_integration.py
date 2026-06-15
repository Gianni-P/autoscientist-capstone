"""Wire MCP-server tools into autoscientist's tool registry.

This is the glue between the transport-level :mod:`autoscientist.clients.mcp_bridge`
and the synchronous :mod:`autoscientist.tools.registry` the agent loop dispatches
against. Given a server key (defined in ``config/mcp.toml``) it:

  1. builds the transport (remote Streamable HTTP, or local stdio subprocess)
     from config + secrets resolved out of the environment,
  2. lazily starts a :class:`~autoscientist.clients.mcp_bridge.MCPServerConnection`
     (cached per key, idempotent across agent invocations in one process),
  3. discovers the server's tools, filters them to the configured allowlist,
     and registers each into the registry under a prefix (e.g. ``github_``)
     with a handler that forwards the call to the MCP server.

The runner calls :func:`ensure_server` before an agent's tool-use loop. Failures
(no token, Docker not reachable, server down) raise :class:`MCPConfigError` /
:class:`MCPBridgeError`; the runner catches them and lets the agent proceed with
its native tools only — publishing to GitHub is best-effort, writing the local
release tree is not.

Config shape (``config/mcp.toml``)::

    [servers.github]
    enabled = true
    transport = "http"                 # "http" (remote) or "stdio" (local docker)
    tool_prefix = "github_"
    allowed_tools = ["get_me", "create_repository", "push_files", ...]
    call_timeout_s = 120
    startup_timeout_s = 60

    [servers.github.http]
    url = "https://api.githubcopilot.com/mcp/"
    token_env = "GITHUB_PERSONAL_ACCESS_TOKEN"
    auth_header = "Authorization"
    auth_value = "Bearer {token}"
    [servers.github.http.extra_headers]
    "X-MCP-Toolsets" = "repos,pull_requests,users"

    [servers.github.stdio]
    command = "docker"
    args = ["run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server"]
    token_env = "GITHUB_PERSONAL_ACCESS_TOKEN"
    env_passthrough = ["PATH", "HOME"]
    toolsets = "repos,pull_requests,users"   # exported as GITHUB_TOOLSETS
"""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
from typing import TYPE_CHECKING, Any

import structlog

from autoscientist.clients.mcp_bridge import (
    HttpTransport,
    MCPServerConnection,
    StdioTransport,
    Transport,
)
from autoscientist.tools import registry as tool_registry

if TYPE_CHECKING:
    from autoscientist.runtime.config import Config

log = structlog.get_logger("autoscientist.tools.mcp_integration")


class MCPConfigError(RuntimeError):
    """The requested MCP server is missing/disabled/misconfigured."""


# Live connections + the prefixed tool names they registered, keyed by server.
_CONNECTIONS: dict[str, MCPServerConnection] = {}
_REGISTERED: dict[str, list[str]] = {}
_LOCK = threading.Lock()


def _server_config(cfg: Config, server_key: str) -> dict[str, Any]:
    servers = (cfg.mcp or {}).get("servers", {})
    sc = servers.get(server_key)
    if sc is None:
        raise MCPConfigError(
            f"no MCP server '{server_key}' in config/mcp.toml "
            f"(known: {sorted(servers)})"
        )
    if not sc.get("enabled", True):
        raise MCPConfigError(f"MCP server '{server_key}' is disabled in config")
    return sc


def _build_transport(server_key: str, sc: dict[str, Any]) -> Transport:
    kind = sc.get("transport", "http")
    if kind == "http":
        h = sc.get("http", {})
        url = h.get("url")
        if not url:
            raise MCPConfigError(f"server '{server_key}' http.url is required")
        headers: dict[str, str] = dict(h.get("extra_headers", {}))
        token = _resolve_token(h.get("token_env"))
        # If a token env var is configured, require it (mirrors the stdio
        # branch). Otherwise removing the auth_value line would silently connect
        # UNauthenticated instead of degrading cleanly.
        if h.get("token_env") and not token:
            raise MCPConfigError(
                f"server '{server_key}': env var {h.get('token_env')!r} is not set"
            )
        auth_value = h.get("auth_value")
        if auth_value:
            headers[h.get("auth_header", "Authorization")] = auth_value.format(
                token=token or ""
            )
        return HttpTransport(url=url, headers=headers or None)

    if kind == "stdio":
        s = sc.get("stdio", {})
        command = s.get("command")
        if not command:
            raise MCPConfigError(f"server '{server_key}' stdio.command is required")
        env: dict[str, str] = {
            v: os.environ[v] for v in s.get("env_passthrough", []) if v in os.environ
        }
        token_env = s.get("token_env")
        if token_env:
            token = _resolve_token(token_env)
            if not token:
                raise MCPConfigError(
                    f"server '{server_key}': env var {token_env!r} is not set"
                )
            env[token_env] = token
        if s.get("toolsets"):
            env["GITHUB_TOOLSETS"] = str(s["toolsets"])
        for k, v in (s.get("extra_env") or {}).items():
            env[k] = str(v)
        return StdioTransport(
            command=command,
            args=tuple(s.get("args", [])),
            env=env or None,
            cwd=s.get("cwd"),
        )

    raise MCPConfigError(f"server '{server_key}': unknown transport {kind!r}")


def _resolve_token(env_var: str | None) -> str | None:
    if not env_var:
        return None
    return os.environ.get(env_var) or None


def _make_handler(server_key: str, real_name: str):
    """Build a registry handler that forwards a call to the MCP server."""

    def handler(inp: dict[str, Any], ctx: tool_registry.ToolContext) -> dict[str, Any]:
        conn = _CONNECTIONS.get(server_key)
        if conn is None or not conn.is_ready:
            raise MCPConfigError(
                f"MCP server '{server_key}' is not connected (tool {real_name!r})"
            )
        # ctx is unused: MCP tools talk to the remote server, not the sandbox.
        return conn.call_tool(real_name, inp)

    return handler


def ensure_server(server_key: str, cfg: Config) -> list[str]:
    """Connect to ``server_key`` (if needed) and register its tools.

    Returns the list of *prefixed* tool names registered for this server.
    Idempotent: a healthy cached connection is reused. Raises
    :class:`MCPConfigError` / :class:`~autoscientist.clients.mcp_bridge.MCPBridgeError`
    on failure — callers that want graceful degradation must catch.
    """
    # The lock is held across conn.start() (which blocks on the handshake, up to
    # startup_timeout_s). That intentionally serializes connection setup: the
    # runner drives agents sequentially in one thread, so there is no contention
    # to optimize away. If a concurrent caller is ever added, switch to a
    # per-server-key lock so independent servers can connect in parallel.
    with _LOCK:
        existing = _CONNECTIONS.get(server_key)
        if existing is not None:
            if existing.is_ready:
                return list(_REGISTERED.get(server_key, []))
            # Stale/dead connection — drop it, retract its now-dead tools, and
            # reconnect below so we never offer an agent a tool whose server is
            # gone.
            with contextlib.suppress(Exception):
                existing.stop()
            _CONNECTIONS.pop(server_key, None)
            for name in _REGISTERED.pop(server_key, []):
                tool_registry.unregister(name)

        sc = _server_config(cfg, server_key)
        transport = _build_transport(server_key, sc)
        conn = MCPServerConnection(
            transport,
            name=server_key,
            call_timeout_s=float(sc.get("call_timeout_s", 120.0)),
            startup_timeout_s=float(sc.get("startup_timeout_s", 60.0)),
        )
        conn.start()  # raises on failure; nothing cached if it does
        _CONNECTIONS[server_key] = conn

        prefix = sc.get("tool_prefix", f"{server_key}_")
        allowed = set(sc.get("allowed_tools", []))
        registered: list[str] = []
        skipped: list[str] = []
        for td in conn.list_tools():
            if allowed and td.name not in allowed:
                skipped.append(td.name)
                continue
            reg_name = f"{prefix}{td.name}"
            tool_registry.register_or_replace(
                tool_registry.ToolSpec(
                    name=reg_name,
                    description=td.description or f"{server_key} tool {td.name}",
                    input_schema=td.input_schema or {"type": "object"},
                    handler=_make_handler(server_key, td.name),
                )
            )
            registered.append(reg_name)
        _REGISTERED[server_key] = registered
        log.info(
            "mcp_integration.server_ready",
            server=server_key,
            transport=sc.get("transport", "http"),
            registered=registered,
            skipped_count=len(skipped),
        )
        return registered


def registered_tools(server_key: str) -> list[str]:
    return list(_REGISTERED.get(server_key, []))


def shutdown_all() -> None:
    """Stop every live MCP connection and retract their tools. Registered at
    process exit; also safe to call between runs in a long-lived process."""
    with _LOCK:
        for key, conn in list(_CONNECTIONS.items()):
            with contextlib.suppress(Exception):
                conn.stop()
            _CONNECTIONS.pop(key, None)
            for name in _REGISTERED.pop(key, []):
                tool_registry.unregister(name)
        _REGISTERED.clear()


atexit.register(shutdown_all)
