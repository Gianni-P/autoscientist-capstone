"""Bridge between autoscientist's *synchronous* tool runtime and *async* MCP servers.

The agent runner (``runtime/runner.py``) and the tool registry
(``tools/registry.py``) are synchronous. The Model Context Protocol Python SDK
(``mcp``) is async and exposes a server connection as nested async context
managers::

    async with stdio_client(params) as (read, write):          # local subprocess
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.list_tools()
            await session.call_tool(name, args)

    # or, for a remote server (no Docker needed):
    async with streamablehttp_client(url, headers=...) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            ...

Two facts force the design below:

  1. A :class:`ClientSession` is only usable while those ``async with`` blocks
     are open. We need it open across *many* ``call_tool`` calls spanning the
     minutes a ``repo_publisher`` invocation takes.
  2. The SDK is built on anyio; its task group and cancel scopes must be
     entered and exited on the **same task**. Driving the contexts from one
     thread and cancelling from another corrupts the cancel-scope stack.

So we host the entire session lifecycle inside ONE coroutine running on a
dedicated event loop in a background daemon thread:

  * the coroutine opens the transport + session, initializes, publishes the
    discovered tool list, then awaits a stop event — keeping the contexts open;
  * synchronous callers submit ``session.call_tool(...)`` onto that loop via
    :func:`asyncio.run_coroutine_threadsafe` and block on the returned future;
  * :meth:`MCPServerConnection.stop` flips the stop event via
    ``loop.call_soon_threadsafe`` so the coroutine unwinds the contexts *in its
    own task* before the loop closes.

Every anyio scope enter/exit therefore happens on the loop thread and never
crosses a thread boundary — the one rule that makes this safe.

This module is pure transport: it knows nothing about autoscientist's registry,
agents, or config. ``tools/mcp_integration.py`` wires discovered tools into the
:class:`~autoscientist.tools.registry.ToolSpec` registry.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import structlog
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

log = structlog.get_logger("autoscientist.clients.mcp_bridge")

# Sensible defaults. A tool call (e.g. pushing a tree of files to GitHub) can
# take a while; the startup handshake should not.
DEFAULT_CALL_TIMEOUT_S = 120.0
DEFAULT_STARTUP_TIMEOUT_S = 60.0
DEFAULT_STOP_TIMEOUT_S = 15.0


class MCPBridgeError(RuntimeError):
    """A bridge-level failure: connection never came up, call timed out, etc."""


@dataclass(frozen=True)
class MCPToolDef:
    """A tool as advertised by an MCP server, in autoscientist-friendly shape."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class StdioTransport:
    """Launch the server as a local subprocess and talk to it over stdio.

    ``env`` is passed verbatim to the child. The MCP SDK does *not* inherit the
    parent environment when ``env`` is set, so callers must include everything
    the child needs (e.g. ``PATH`` plus the server's auth token).
    """

    command: str
    args: tuple[str, ...] = ()
    # repr=False: env may carry an auth token (e.g. GITHUB_PERSONAL_ACCESS_TOKEN)
    # — keep it out of reprs/logs/tracebacks.
    env: dict[str, str] | None = field(default=None, repr=False)
    cwd: str | None = None


@dataclass(frozen=True)
class HttpTransport:
    """Connect to a remote server over Streamable HTTP (no subprocess)."""

    url: str
    # repr=False: headers carry the Authorization bearer token — keep it out of
    # reprs/logs/tracebacks.
    headers: dict[str, str] | None = field(default=None, repr=False)


Transport = StdioTransport | HttpTransport


@dataclass
class MCPServerConnection:
    """A live, synchronously-callable connection to one MCP server.

    Usage::

        conn = MCPServerConnection(transport, name="github")
        conn.start()                       # blocks until initialized
        for t in conn.list_tools(): ...
        out = conn.call_tool("create_repository", {"name": "demo"})
        conn.stop()

    or as a context manager::

        with MCPServerConnection(transport, name="github") as conn:
            conn.call_tool(...)
    """

    # repr=False: the transport carries auth secrets (env token / bearer header).
    transport: Transport = field(repr=False)
    name: str = "mcp"
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S

    # --- internal state (not init args) ---
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _loop: asyncio.AbstractEventLoop | None = field(default=None, init=False, repr=False)
    _session: ClientSession | None = field(default=None, init=False, repr=False)
    _stop_event: asyncio.Event | None = field(default=None, init=False, repr=False)
    _ready: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _ready_ok: bool = field(default=False, init=False, repr=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _tools: list[MCPToolDef] = field(default_factory=list, init=False, repr=False)
    _server_info: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ start
    def start(self) -> None:
        """Spawn the loop thread, open the session, discover tools.

        Blocks until the server has initialized (``startup_timeout_s``). Raises
        :class:`MCPBridgeError` if the handshake fails or times out.
        """
        if self._thread is not None:
            raise MCPBridgeError(f"connection '{self.name}' already started")
        self._thread = threading.Thread(
            target=self._run_loop, name=f"mcp-{self.name}", daemon=True
        )
        self._thread.start()
        # The loop coroutine sets _ready on either success or failure. On any
        # failure, stop() before raising so we never leak the in-flight thread /
        # subprocess and the object is left clean and restartable.
        if not self._ready.wait(timeout=self.startup_timeout_s + 5.0):
            self.stop()
            raise MCPBridgeError(
                f"MCP server '{self.name}' did not initialize within "
                f"{self.startup_timeout_s}s"
            )
        if not self._ready_ok:
            err = self._error
            self.stop()
            raise MCPBridgeError(f"MCP server '{self.name}' failed to start: {err!r}")
        log.info(
            "mcp_bridge.started",
            server=self.name,
            tool_count=len(self._tools),
            server_info=self._server_info,
        )

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except Exception as e:  # pragma: no cover - belt and braces
            self._error = e
            self._ready_ok = False
            self._ready.set()
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def _open_transport(self):
        """Return the async context manager for the configured transport."""
        t = self.transport
        if isinstance(t, StdioTransport):
            params = StdioServerParameters(
                command=t.command,
                args=list(t.args),
                env=t.env,
                cwd=t.cwd,
            )
            return stdio_client(params)
        if isinstance(t, HttpTransport):
            return streamablehttp_client(t.url, headers=t.headers)
        raise MCPBridgeError(f"unsupported transport: {type(t).__name__}")

    async def _serve(self) -> None:
        """Own the session for its whole lifetime on this loop's single task."""
        self._stop_event = asyncio.Event()
        try:
            async with self._open_transport() as streams:
                # stdio yields (read, write); streamable-http yields
                # (read, write, get_session_id) — take the first two either way.
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    init = await asyncio.wait_for(
                        session.initialize(), timeout=self.startup_timeout_s
                    )
                    listed = await session.list_tools()
                    self._session = session
                    self._tools = [
                        MCPToolDef(
                            name=tool.name,
                            description=tool.description or "",
                            input_schema=tool.inputSchema or {"type": "object"},
                        )
                        for tool in listed.tools
                    ]
                    server = getattr(init, "serverInfo", None)
                    if server is not None:
                        self._server_info = {
                            "name": getattr(server, "name", None),
                            "version": getattr(server, "version", None),
                        }
                    self._ready_ok = True
                    self._ready.set()
                    # Hold the contexts open until stop() is requested.
                    await self._stop_event.wait()
        except Exception as e:
            self._error = e
            self._ready_ok = False
            log.warning(
                "mcp_bridge.serve_error",
                server=self.name,
                error=str(e),
                error_type=type(e).__name__,
            )
            # Unblock start() even on failure.
            self._ready.set()
        finally:
            self._session = None

    # ------------------------------------------------------------- discovery
    def list_tools(self) -> list[MCPToolDef]:
        """Tools advertised by the server (captured at initialize time)."""
        return list(self._tools)

    @property
    def server_info(self) -> dict[str, Any]:
        return dict(self._server_info)

    @property
    def is_ready(self) -> bool:
        return self._ready_ok and self._session is not None

    # ------------------------------------------------------------- call_tool
    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Invoke a tool and return a normalized JSON-safe dict.

        Returns ``{"is_error": bool, "text": str, "structured": <obj|None>}``.
        ``structured`` is the server's ``structuredContent`` when present, else
        the parsed JSON of the text payload when it parses, else ``None``.

        Raises :class:`MCPBridgeError` if the connection is down or the call
        does not return within ``timeout_s``.
        """
        loop = self._loop
        session = self._session
        if loop is None or session is None or not self._ready_ok:
            raise MCPBridgeError(
                f"MCP server '{self.name}' is not connected (call to {name!r})"
            )
        timeout = timeout_s if timeout_s is not None else self.call_timeout_s
        coro = session.call_tool(
            name,
            arguments or {},
            read_timeout_seconds=timedelta(seconds=timeout),
        )
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            # Outer guard slightly beyond the protocol read-timeout so the
            # server-side timeout fires first and gives a cleaner error.
            result = future.result(timeout=timeout + 10.0)
        except FuturesTimeout as e:
            future.cancel()
            raise MCPBridgeError(
                f"MCP call '{self.name}.{name}' timed out after {timeout}s"
            ) from e
        except Exception as e:
            # Anything raised inside the coroutine — the protocol read-timeout
            # (mcp.McpError REQUEST_TIMEOUT, which fires ~10s before the outer
            # guard), transport errors, etc. — is re-wrapped so the bridge's
            # documented MCPBridgeError contract holds for every failure path.
            # Tool-level failures don't land here: they return a CallToolResult
            # with isError=True, handled by _normalize_result.
            raise MCPBridgeError(f"MCP call '{self.name}.{name}' failed: {e}") from e
        return _normalize_result(result)

    # ------------------------------------------------------------------ stop
    def stop(self, timeout_s: float = DEFAULT_STOP_TIMEOUT_S) -> None:
        """Signal the loop coroutine to unwind and join the thread."""
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            # RuntimeError here just means the loop already stopped — ignore it.
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(stop_event.set)
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                # Don't drop the handle: a wedged transport (e.g. a hung Docker
                # container) means the thread/subprocess is still alive. Keeping
                # _thread lets a later stop()/shutdown_all() re-attempt the join
                # instead of orphaning the thread, loop, and subprocess forever.
                log.warning("mcp_bridge.stop_timeout", server=self.name)
                return
            self._thread = None
        log.info("mcp_bridge.stopped", server=self.name)

    # --------------------------------------------------------- context mgr
    def __enter__(self) -> MCPServerConnection:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _normalize_result(result: Any) -> dict[str, Any]:
    """Flatten a ``CallToolResult`` into a JSON-safe dict for the LLM loop."""
    texts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text is not None:
            texts.append(text)
    joined = "\n".join(texts)

    structured = getattr(result, "structuredContent", None)
    if structured is None and joined:
        try:
            structured = json.loads(joined)
        except (ValueError, TypeError):
            structured = None

    return {
        "is_error": bool(getattr(result, "isError", False)),
        "text": joined,
        "structured": structured,
    }
