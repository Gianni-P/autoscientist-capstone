"""Shared SQLite-backed cache for tool results.

Each tool gets its own namespace via ``tool_name``. Keys are SHA256 hashes
of canonicalized JSON-serializable inputs. Values are JSON-serializable
payloads stored as text. Hits don't cost anything; misses run the real tool.

Pattern in tool modules:

    from autoscientist.tools import tool_cache

    key = tool_cache.cache_key({"q": query, "limit": limit})
    cached = tool_cache.cache_get(conn, "literature.search", key)
    if cached is not None:
        return cached
    result = run_tool(query)
    tool_cache.cache_put(conn, "literature.search", key, result)
    return result
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from autoscientist.state.db import now_iso


def _canonicalize(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def cache_key(payload: Any) -> str:
    return hashlib.sha256(_canonicalize(payload).encode("utf-8")).hexdigest()


def cache_get(conn: sqlite3.Connection, tool_name: str, key: str) -> Any | None:
    row = conn.execute(
        "SELECT payload FROM tool_cache WHERE tool_name = ? AND cache_key = ?",
        (tool_name, key),
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE tool_cache SET hit_count = hit_count + 1, last_hit_at = ? "
        "WHERE tool_name = ? AND cache_key = ?",
        (now_iso(), tool_name, key),
    )
    conn.commit()
    return json.loads(row["payload"])


def cache_put(conn: sqlite3.Connection, tool_name: str, key: str, payload: Any) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO tool_cache "
        "(tool_name, cache_key, payload, created_at, last_hit_at, hit_count) "
        "VALUES (?, ?, ?, ?, NULL, "
        "  COALESCE((SELECT hit_count FROM tool_cache "
        "            WHERE tool_name = ? AND cache_key = ?), 0))",
        (tool_name, key, _canonicalize(payload), now_iso(), tool_name, key),
    )
    conn.commit()
