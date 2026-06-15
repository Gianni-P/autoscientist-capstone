"""SQLite schema and thin accessors for autoscientist state.

Tables
------
  runs            One row per pipeline run.
  messages        Every prompt/response/handoff in the agent loop.
  cache           SHA256-keyed response cache (free hits, paid misses).
  budget_ledger   Append-only spend ledger with month_key for fast aggregation.
  checkpoints     Human-in-the-loop gates with payload + operator response.

This module owns the schema and the connection. Higher-level modules
(clients/cache.py, runtime/budget.py, runtime/runner.py) acquire a
connection via :func:`get_db` and execute their own SQL.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('running','paused','completed','failed','cancelled')),
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    config_snapshot  TEXT,
    note             TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_runs_status  ON runs(status);

CREATE TABLE IF NOT EXISTS messages (
    message_id        TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    agent_name        TEXT NOT NULL,
    role              TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool','handoff')),
    content           TEXT NOT NULL,
    reasoning         TEXT,
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cost_usd          REAL,
    latency_ms        INTEGER,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_run   ON messages(run_id);
CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_name);

CREATE TABLE IF NOT EXISTS cache (
    cache_key         TEXT PRIMARY KEY,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    request_blob      TEXT NOT NULL,
    response_blob     TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    created_at        TEXT NOT NULL,
    hit_count         INTEGER NOT NULL DEFAULT 0,
    last_hit_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_cache_model ON cache(model);

CREATE TABLE IF NOT EXISTS budget_ledger (
    ledger_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT,
    agent_name        TEXT,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    cache_hit         INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    month_key         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_budget_month ON budget_ledger(month_key);
CREATE INDEX IF NOT EXISTS idx_budget_run   ON budget_ledger(run_id);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id   TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    stage           INTEGER NOT NULL CHECK (stage BETWEEN 1 AND 5),
    payload         TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending','approved','rejected','modified')),
    operator_input  TEXT,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run    ON checkpoints(run_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON checkpoints(status);

CREATE TABLE IF NOT EXISTS tool_cache (
    tool_name      TEXT NOT NULL,
    cache_key      TEXT NOT NULL,
    payload        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    last_hit_at    TEXT,
    hit_count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tool_name, cache_key)
);

CREATE INDEX IF NOT EXISTS idx_tool_cache_tool ON tool_cache(tool_name);

CREATE TABLE IF NOT EXISTS checkpoint_questions (
    question_id    TEXT PRIMARY KEY,
    checkpoint_id  TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role IN ('operator','assistant')),
    content        TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    agent_used     TEXT,
    cost_usd       REAL,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_cpq_checkpoint ON checkpoint_questions(checkpoint_id);

CREATE TABLE IF NOT EXISTS prompt_versions (
    version_id        TEXT PRIMARY KEY,
    agent_name        TEXT NOT NULL,
    prompt_text       TEXT NOT NULL,
    parent_version_id TEXT,
    note              TEXT,
    archived_path     TEXT,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pv_agent ON prompt_versions(agent_name);

CREATE TABLE IF NOT EXISTS eval_runs (
    eval_run_id        TEXT PRIMARY KEY,
    agent_name         TEXT NOT NULL,
    prompt_version_id  TEXT NOT NULL,
    anchor_id          TEXT NOT NULL,
    raw_output         TEXT,
    rubric_scores      TEXT,
    total_score        REAL,
    judge_model        TEXT,
    judge_cost_usd     REAL,
    judge_summary      TEXT,
    note               TEXT,
    created_at         TEXT NOT NULL,
    FOREIGN KEY (prompt_version_id) REFERENCES prompt_versions(version_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_runs_agent ON eval_runs(agent_name);
CREATE INDEX IF NOT EXISTS idx_eval_runs_version ON eval_runs(prompt_version_id);

-- Manual pause / resume state. One row per run when the operator has either
-- requested a pause or successfully paused the chain between agents. Distinct
-- from `checkpoints` (which are KICKOFF §7 HITL gates) — these are operator
-- breaks the runner honours at the next safe boundary.
CREATE TABLE IF NOT EXISTS run_controls (
    run_id              TEXT PRIMARY KEY,
    pause_requested     INTEGER NOT NULL DEFAULT 0,
    requested_at        TEXT,
    paused_at           TEXT,
    next_agent          TEXT,
    next_payload        TEXT,
    handoffs_so_far     INTEGER,
    code_review_cycles  INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_run_controls_pause
    ON run_controls(pause_requested);
"""


def now_iso() -> str:
    # Millisecond precision so ORDER BY created_at sorts events that
    # happen within the same second (e.g. multiple tool-loop rounds).
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def month_key(ts: datetime | None = None) -> str:
    ts = ts or datetime.now(UTC)
    return ts.strftime("%Y-%m")


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex
    return f"{prefix}{raw}" if prefix else raw


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    elif row["version"] < SCHEMA_VERSION:
        # All schema additions to date are pure CREATE TABLE IF NOT EXISTS, so
        # the executescript above already added any missing tables for older
        # DBs. Just bump the recorded version.
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    conn.commit()


def open_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with WAL mode + foreign keys + 30s busy timeout.

    Auto-creates the schema on first open. Idempotent.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    _ensure_schema(conn)
    return conn


@contextmanager
def get_db(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Context-managed connection. Commits on clean exit, rolls back on error."""
    conn = open_db(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run lifecycle helpers
# ---------------------------------------------------------------------------

def start_run(
    conn: sqlite3.Connection,
    project_id: str,
    config_snapshot: dict[str, Any] | None = None,
    note: str | None = None,
) -> str:
    run_id = new_id("run_")
    conn.execute(
        "INSERT INTO runs (run_id, project_id, status, started_at, config_snapshot, note) "
        "VALUES (?, ?, 'running', ?, ?, ?)",
        (run_id, project_id, now_iso(), json.dumps(config_snapshot) if config_snapshot else None, note),
    )
    return run_id


def end_run(conn: sqlite3.Connection, run_id: str, status: str, note: str | None = None) -> None:
    if status not in {"completed", "failed", "cancelled", "paused"}:
        raise ValueError(f"invalid run status: {status}")
    conn.execute(
        "UPDATE runs SET status = ?, ended_at = ?, note = COALESCE(?, note) WHERE run_id = ?",
        (status, now_iso(), note, run_id),
    )


def record_message(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    agent_name: str,
    role: str,
    content: str,
    reasoning: str | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
    latency_ms: int | None = None,
    cache_hit: bool = False,
) -> str:
    message_id = new_id("msg_")
    conn.execute(
        """INSERT INTO messages (
            message_id, run_id, agent_name, role, content, reasoning, model,
            prompt_tokens, completion_tokens, cost_usd, latency_ms, cache_hit, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            message_id, run_id, agent_name, role, content, reasoning, model,
            prompt_tokens, completion_tokens, cost_usd, latency_ms, int(cache_hit), now_iso(),
        ),
    )
    return message_id
