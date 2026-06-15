"""SHA256-keyed response cache for LLM calls.

Cache hits are free; misses cost real money. Callers (router.py) must
check the cache before making a network call. The cache is keyed over
the *canonical* request — provider, model, system prompt, message list,
temperature, max_tokens, and any tools signature — so two identical
requests collapse to one paid call regardless of which run issued them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from autoscientist.state.db import now_iso


@dataclass
class CacheEntry:
    cache_key: str
    provider: str
    model: str
    request_blob: dict[str, Any]
    response_blob: dict[str, Any]
    prompt_tokens: int | None
    completion_tokens: int | None
    hit_count: int


def _canonicalize(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(
    *,
    provider: str,
    model: str,
    system: str | None,
    messages: list[dict[str, Any]],
    temperature: float | None,
    max_tokens: int | None,
    tools_signature: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "system": system or "",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools_signature": tools_signature,
        "extra": extra or {},
    }
    return hashlib.sha256(_canonicalize(payload).encode("utf-8")).hexdigest()


def get_cached(conn: sqlite3.Connection, key: str) -> CacheEntry | None:
    row = conn.execute(
        "SELECT cache_key, provider, model, request_blob, response_blob, "
        "prompt_tokens, completion_tokens, hit_count FROM cache WHERE cache_key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE cache SET hit_count = hit_count + 1, last_hit_at = ? WHERE cache_key = ?",
        (now_iso(), key),
    )
    return CacheEntry(
        cache_key=row["cache_key"],
        provider=row["provider"],
        model=row["model"],
        request_blob=json.loads(row["request_blob"]),
        response_blob=json.loads(row["response_blob"]),
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        hit_count=row["hit_count"] + 1,
    )


def put_cached(
    conn: sqlite3.Connection,
    *,
    key: str,
    provider: str,
    model: str,
    request_blob: dict[str, Any],
    response_blob: dict[str, Any],
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO cache (
            cache_key, provider, model, request_blob, response_blob,
            prompt_tokens, completion_tokens, created_at, hit_count, last_hit_at
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?,
            COALESCE((SELECT hit_count FROM cache WHERE cache_key = ?), 0),
            NULL
        )""",
        (
            key, provider, model,
            _canonicalize(request_blob), _canonicalize(response_blob),
            prompt_tokens, completion_tokens, now_iso(), key,
        ),
    )
