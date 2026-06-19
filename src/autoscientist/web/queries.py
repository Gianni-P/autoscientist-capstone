"""Read layer for the web console: SQLite rows → plain JSON-able dicts.

Every function here is a pure read (no writes) and returns ``dict``/``list``
structures the frontend consumes directly. Mutations (resolve / pause /
resume / ask) go through ``cp_manager`` and ``runtime.control`` in
``app.py`` — this module never writes.

Cursor model
~~~~~~~~~~~~
``messages`` keeps its implicit ``rowid`` (the table is not WITHOUT ROWID),
which is monotonic with insert order. We use it as the incremental cursor
for the live feed: the SSE loop tracks ``max(rowid)`` globally and the
client fetches ``rowid > cursor`` for whichever run it is viewing.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.runtime import control as run_control
from autoscientist.runtime.budget import BudgetConfig, monthly_spent
from autoscientist.runtime.config import load_config
from autoscientist.state.db import month_key, open_db

# A generous server-side cap on a single content blob shipped in a list
# response. Anything larger is truncated with a flag; the full text is
# available via ``GET /api/messages/{message_id}``.
_CONTENT_CAP = 16_000

_STATUS_DOT = {
    "running": "running",
    "paused": "paused",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
}


def connect() -> sqlite3.Connection:
    """Open a read connection on the configured DB (WAL → concurrent reads)."""
    return open_db(load_config().db_path())


def _d(row: sqlite3.Row) -> dict[str, Any]:
    # sqlite3.Row iterates its *values*, not keys — .keys() is required here.
    return {k: row[k] for k in row.keys()}  # noqa: SIM118


def _cap(text: str | None) -> tuple[str, bool]:
    text = text or ""
    if len(text) > _CONTENT_CAP:
        return text[:_CONTENT_CAP], True
    return text, False


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def budget_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    cfg = load_config()
    bcfg = BudgetConfig.from_dict(cfg.models.get("budget", {}))
    spent = monthly_spent(conn)
    cap = float(bcfg.monthly_cap_usd)
    return {
        "month": month_key(),
        "spent": round(spent, 4),
        "cap": round(cap, 2),
        "pct": min(1.0, spent / cap) if cap else 0.0,
        "remaining": round(max(0.0, cap - spent), 4),
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def list_runs(conn: sqlite3.Connection, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT run_id, project_id, status, started_at, ended_at, note "
        "FROM runs ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    runs = [_d(r) for r in rows]
    if not runs:
        return runs
    ids = [r["run_id"] for r in runs]
    placeholders = ",".join("?" * len(ids))
    agg = {
        r["run_id"]: r
        for r in conn.execute(
            f"SELECT run_id, COUNT(*) AS messages, "
            f"COALESCE(SUM(cost_usd),0) AS cost, MAX(created_at) AS last_at "
            f"FROM messages WHERE run_id IN ({placeholders}) GROUP BY run_id",
            ids,
        ).fetchall()
    }
    # Latest agent per run (the "what's running" hint on the list).
    last_agent: dict[str, str] = {}
    for r in conn.execute(
        f"SELECT m.run_id, m.agent_name FROM messages m "
        f"JOIN (SELECT run_id, MAX(rowid) AS mx FROM messages "
        f"      WHERE run_id IN ({placeholders}) GROUP BY run_id) t "
        f"ON m.run_id = t.run_id AND m.rowid = t.mx",
        ids,
    ).fetchall():
        last_agent[r["run_id"]] = r["agent_name"]
    for r in runs:
        a = agg.get(r["run_id"])
        r["message_count"] = a["messages"] if a else 0
        r["total_cost"] = round(a["cost"], 4) if a else 0.0
        r["last_activity"] = a["last_at"] if a else r["started_at"]
        r["current_agent"] = last_agent.get(r["run_id"])
    return runs


def _stage_rollup(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """The five HITL stages with the latest checkpoint status for each."""
    cps = cp_manager.list_for_run(conn, run_id)
    latest_by_stage: dict[int, cp_manager.CheckpointRecord] = {}
    for cp in cps:  # list_for_run is created_at ASC → last write wins
        latest_by_stage[cp.stage] = cp
    stages: list[dict[str, Any]] = []
    for stage in (1, 2, 3, 4, 5):
        cp = latest_by_stage.get(stage)
        stages.append(
            {
                "stage": stage,
                "title": cp_manager.STAGE_TITLES.get(stage, f"Stage {stage}"),
                "name": cp_manager.STAGE_NAMES.get(stage, f"stage_{stage}"),
                "status": cp.status if cp else "todo",  # pending|approved|modified|rejected|todo
                "checkpoint_id": cp.checkpoint_id if cp else None,
                "created_at": cp.created_at if cp else None,
                "resolved_at": cp.resolved_at if cp else None,
                "loop_cap": bool((cp.extra or {}).get("loop_cap_exceeded")) if cp else False,
            }
        )
    return stages


def run_detail(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    run = _d(row)

    stages = _stage_rollup(conn, run_id)
    pending = next((s for s in stages if s["status"] == "pending"), None)
    # current stage = the pending one, else the furthest resolved stage.
    resolved = [s for s in stages if s["status"] in {"approved", "modified", "rejected"}]
    current_stage = pending["stage"] if pending else (resolved[-1]["stage"] if resolved else None)

    last_msg = conn.execute(
        "SELECT agent_name, role, content, created_at, model, cost_usd "
        "FROM messages WHERE run_id = ? ORDER BY rowid DESC LIMIT 1",
        (run_id,),
    ).fetchone()

    cost_row = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost, "
        "COALESCE(SUM(prompt_tokens),0) AS pt, COALESCE(SUM(completion_tokens),0) AS ct "
        "FROM messages WHERE run_id = ?",
        (run_id,),
    ).fetchone()

    pause = run_control.read_pause_state(conn, run_id)
    return {
        "run": run,
        "stages": stages,
        "current_stage": current_stage,
        "pending_checkpoint": pending,
        "current_agent": last_msg["agent_name"] if last_msg else None,
        "last_event": (
            {
                "agent": last_msg["agent_name"],
                "role": last_msg["role"],
                "preview": _short(last_msg["content"]),
                "at": last_msg["created_at"],
            }
            if last_msg
            else None
        ),
        "totals": {
            "messages": cost_row["n"],
            "cost": round(cost_row["cost"], 4),
            "prompt_tokens": cost_row["pt"],
            "completion_tokens": cost_row["ct"],
        },
        "pause": _pause_dict(pause),
        "agents": _agent_roster(conn, run_id),
    }


def _pause_dict(pause: run_control.PauseState | None) -> dict[str, Any] | None:
    if pause is None:
        return None
    return {
        "pause_requested": pause.pause_requested,
        "requested_at": pause.requested_at,
        "paused_at": pause.paused_at,
        "next_agent": pause.next_agent,
        "handoffs_so_far": pause.handoffs_so_far,
        "is_active": pause.is_active,
    }


def _agent_roster(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Distinct agents seen in this run, in first-seen order, with counts."""
    rows = conn.execute(
        "SELECT agent_name, COUNT(*) AS n, MIN(rowid) AS first_seen, "
        "MAX(created_at) AS last_at, COALESCE(SUM(cost_usd),0) AS cost "
        "FROM messages WHERE run_id = ? GROUP BY agent_name ORDER BY first_seen ASC",
        (run_id,),
    ).fetchall()
    return [
        {
            "agent": r["agent_name"],
            "events": r["n"],
            "last_at": r["last_at"],
            "cost": round(r["cost"], 4),
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Messages (live feed)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _short(text: str | None, limit: int = 240) -> str:
    if not text:
        return ""
    flat = _WS.sub(" ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
    """Normalize one message row into a feed event the frontend renders."""
    role = row["role"]
    content = row["content"] or ""
    capped, truncated = _cap(content)
    ev: dict[str, Any] = {
        "id": row["message_id"],
        "rowid": row["rowid"],
        "run_id": row["run_id"],
        "agent": row["agent_name"],
        "role": role,
        "at": row["created_at"],
        "preview": _short(content),
        "content": capped,
        "truncated": truncated,
        "model": row["model"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "cost": row["cost_usd"],
        "latency_ms": row["latency_ms"],
        "cache_hit": bool(row["cache_hit"]),
    }
    if role == "tool":
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            data = {}
        ev["tool"] = {
            "name": data.get("name"),
            "duration_ms": data.get("duration_ms", row["latency_ms"]),
            "error": data.get("error"),
            "input": data.get("input"),
            "ok": not data.get("error"),
        }
        ev["preview"] = _short(_tool_input_preview(data.get("input")))
    elif role == "handoff":
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            data = {}
        ev["handoff"] = {
            "checkpoint_id": data.get("checkpoint_id"),
            "decision": data.get("decision"),
            "next_agent": data.get("next_agent"),
        }
    return ev


def _tool_input_preview(inp: Any) -> str:
    if isinstance(inp, dict):
        return ", ".join(
            f"{k}={v if isinstance(v, str) else json.dumps(v)}" for k, v in inp.items()
        )
    if inp is None:
        return ""
    return json.dumps(inp)[:200]


def run_messages(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    after_rowid: int = 0,
    limit: int = 400,
) -> dict[str, Any]:
    """Feed slice for a run.

    ``after_rowid > 0`` returns the *appends* since that cursor (oldest→newest)
    — the live-tail path. ``after_rowid == 0`` (first load) returns the most
    recent ``limit`` events instead of the first ``limit``, so the operator
    lands on current activity rather than ancient history.
    """
    if after_rowid > 0:
        rows = conn.execute(
            "SELECT rowid, message_id, run_id, agent_name, role, content, model, "
            "prompt_tokens, completion_tokens, cost_usd, latency_ms, cache_hit, created_at "
            "FROM messages WHERE run_id = ? AND rowid > ? ORDER BY rowid ASC LIMIT ?",
            (run_id, after_rowid, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT rowid, message_id, run_id, agent_name, role, content, model, "
            "prompt_tokens, completion_tokens, cost_usd, latency_ms, cache_hit, created_at "
            "FROM messages WHERE run_id = ? ORDER BY rowid DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
        rows = list(reversed(rows))
    events = [_event_from_row(r) for r in rows]
    cursor = events[-1]["rowid"] if events else after_rowid
    return {"events": events, "cursor": cursor}


def message_detail(conn: sqlite3.Connection, message_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT rowid, * FROM messages WHERE message_id = ?", (message_id,)
    ).fetchone()
    if row is None:
        return None
    ev = _event_from_row(row)
    ev["content"] = row["content"] or ""  # full, uncapped
    ev["truncated"] = False
    ev["reasoning"] = row["reasoning"]
    return ev


# ---------------------------------------------------------------------------
# Timeline — "each handoff + the prompt for each agent"
# ---------------------------------------------------------------------------

def run_timeline(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Group a run's messages into per-agent activations.

    Each activation captures the *inbound prompt* delivered to an agent
    (the ``role='user'`` payload — the dynamic half of "the prompt for
    each agent"; the static half is the system prompt at
    ``GET /api/agents/{name}/prompt``), the agent's tool calls and final
    output, and the ``HANDOFF`` decision that routed to the next agent.
    """
    rows = conn.execute(
        "SELECT rowid, message_id, agent_name, role, content, model, "
        "prompt_tokens, completion_tokens, cost_usd, latency_ms, created_at "
        "FROM messages WHERE run_id = ? ORDER BY rowid ASC",
        (run_id,),
    ).fetchall()

    acts: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    def _new(agent: str, at: str) -> dict[str, Any]:
        return {
            "seq": len(acts) + 1,
            "agent": agent,
            "started_at": at,
            "ended_at": at,
            "inbound_prompt": None,
            "inbound_message_id": None,
            "inbound_truncated": False,
            "output": None,
            "output_message_id": None,
            "assistant_turns": 0,
            "tool_calls": [],
            "tokens": 0,
            "cost": 0.0,
            "handoff": None,
        }

    for r in rows:
        role = r["role"]
        agent = r["agent_name"]
        if role == "user":
            cur = _new(agent, r["created_at"])
            acts.append(cur)
            prompt, trunc = _cap(r["content"])
            cur["inbound_prompt"] = prompt
            cur["inbound_truncated"] = trunc
            cur["inbound_message_id"] = r["message_id"]
            continue
        if cur is None or (role in {"assistant", "tool"} and agent != cur["agent"]):
            # Activation with no recorded inbound (e.g. resumed mid-chain).
            cur = _new(agent, r["created_at"])
            acts.append(cur)
        cur["ended_at"] = r["created_at"]
        if role == "assistant":
            cur["assistant_turns"] += 1
            cur["tokens"] += (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
            cur["cost"] += r["cost_usd"] or 0.0
            out, _ = _cap(r["content"])
            cur["output"] = out
            cur["output_message_id"] = r["message_id"]
        elif role == "tool":
            try:
                data = json.loads(r["content"] or "{}")
            except (json.JSONDecodeError, TypeError):
                data = {}
            cur["tool_calls"].append(
                {
                    "name": data.get("name", "?"),
                    "ok": not data.get("error"),
                    "duration_ms": data.get("duration_ms", r["latency_ms"]),
                }
            )
        elif role == "handoff":
            try:
                data = json.loads(r["content"] or "{}")
            except (json.JSONDecodeError, TypeError):
                data = {}
            cur["handoff"] = {
                "decision": data.get("decision"),
                "next_agent": data.get("next_agent"),
                "checkpoint_id": data.get("checkpoint_id"),
            }
    for a in acts:
        a["cost"] = round(a["cost"], 4)
    return acts


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_detail(conn: sqlite3.Connection, cp_id: str) -> dict[str, Any] | None:
    cp = cp_manager.get_checkpoint(conn, cp_id)
    if cp is None:
        return None
    questions = [
        {
            "id": q.question_id,
            "role": q.role,
            "content": q.content,
            "at": q.created_at,
            "agent_used": q.agent_used,
            "cost": q.cost_usd,
        }
        for q in cp_manager.list_questions(conn, cp_id)
    ]
    return {
        "checkpoint_id": cp.checkpoint_id,
        "run_id": cp.run_id,
        "stage": cp.stage,
        "title": cp_manager.STAGE_TITLES.get(cp.stage, f"Stage {cp.stage}"),
        "stage_name": cp.stage_name,
        "status": cp.status,
        "from_agent": cp.from_agent,
        "to_agent": cp.to_agent or "DONE",
        "created_at": cp.created_at,
        "resolved_at": cp.resolved_at,
        "parsed": cp.parsed,
        "raw": cp.payload.get("agent_output_raw", ""),
        "default_payload": cp.default_payload,
        "summary": cp.payload.get("summary"),
        "extra": cp.extra,
        "operator_input": cp.operator_input,
        "questions": questions,
    }


def run_checkpoints(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Every checkpoint for a run, oldest→newest (the 'go back' history)."""
    out = []
    for cp in cp_manager.list_for_run(conn, run_id):
        op = cp.operator_input or {}
        out.append(
            {
                "checkpoint_id": cp.checkpoint_id,
                "stage": cp.stage,
                "title": cp_manager.STAGE_TITLES.get(cp.stage, f"Stage {cp.stage}"),
                "status": cp.status,
                "decision": op.get("decision"),
                "from_agent": cp.from_agent,
                "to_agent": cp.to_agent or "DONE",
                "created_at": cp.created_at,
                "resolved_at": cp.resolved_at,
                "loop_cap": bool((cp.extra or {}).get("loop_cap_exceeded")),
            }
        )
    return out


def list_pending(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    out = []
    for cp in cp_manager.list_pending(conn):
        out.append(
            {
                "checkpoint_id": cp.checkpoint_id,
                "run_id": cp.run_id,
                "stage": cp.stage,
                "title": cp_manager.STAGE_TITLES.get(cp.stage, f"Stage {cp.stage}"),
                "from_agent": cp.from_agent,
                "to_agent": cp.to_agent or "DONE",
                "created_at": cp.created_at,
                "loop_cap": bool((cp.extra or {}).get("loop_cap_exceeded")),
                "extra": cp.extra,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Agent system prompts (the static half of "the prompt for each agent")
# ---------------------------------------------------------------------------

_AGENT_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


def agent_system_prompt(name: str) -> dict[str, Any] | None:
    """Read ``prompts/<name>.md``. Path-traversal-safe (name allowlist)."""
    if not _AGENT_RE.match(name or ""):
        return None
    pdir = load_config().prompts_dir().resolve()
    path = (pdir / f"{name}.md").resolve()
    if pdir not in path.parents or not path.exists():
        return None
    return {"agent": name, "system_prompt": path.read_text(encoding="utf-8")}


# ---------------------------------------------------------------------------
# Projects (for launching new runs from the UI)
# ---------------------------------------------------------------------------

_PROJECT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


def _projects_dir() -> Path:
    cfg = load_config()
    rel = cfg.default.get("paths", {}).get("projects_dir", "projects")
    return (cfg.root / rel).resolve()


def list_projects() -> list[dict[str, Any]]:
    """Project dirs that carry a ``config.toml`` (i.e. real, launchable runs)."""
    pdir = _projects_dir()
    out: list[dict[str, Any]] = []
    if not pdir.exists():
        return out
    for d in sorted(pdir.iterdir()):
        if not d.is_dir() or not (d / "config.toml").exists():
            continue
        desc = ""
        try:
            data = tomllib.loads((d / "config.toml").read_text(encoding="utf-8"))
            desc = (data.get("project", {}) or {}).get("description", "")
        except Exception:
            pass
        out.append(
            {
                "id": d.name,
                "description": desc,
                "has_payload": (d / "kickoff_payload.json").exists(),
            }
        )
    return out


def project_payload(project_id: str) -> str | None:
    """Read ``projects/<id>/kickoff_payload.json``. Path-traversal-safe."""
    if not _PROJECT_RE.match(project_id or ""):
        return None
    pdir = _projects_dir()
    d = (pdir / project_id).resolve()
    if d.parent != pdir:  # reject traversal / nesting
        return None
    payload = d / "kickoff_payload.json"
    return payload.read_text(encoding="utf-8") if payload.exists() else None


def project_exists(project_id: str) -> bool:
    if not _PROJECT_RE.match(project_id or ""):
        return False
    pdir = _projects_dir()
    d = (pdir / project_id).resolve()
    return d.parent == pdir and (d / "config.toml").exists()


# ---------------------------------------------------------------------------
# Console overview (initial snapshot for the landing view)
# ---------------------------------------------------------------------------

def latest_active_run(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT run_id FROM runs WHERE status IN ('running','paused') "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    return row["run_id"] if row else None


def overview(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "budget": budget_snapshot(conn),
        "runs": list_runs(conn, limit=60),
        "pending": list_pending(conn),
        "active_run_id": latest_active_run(conn),
    }


# ---------------------------------------------------------------------------
# Change signatures (drive the SSE push loop)
# ---------------------------------------------------------------------------

def change_signature(conn: sqlite3.Connection) -> dict[str, Any]:
    """A cheap snapshot the SSE loop diffs to decide what to push."""
    max_rowid = conn.execute("SELECT COALESCE(MAX(rowid),0) AS m FROM messages").fetchone()["m"]
    runs_sig = conn.execute(
        "SELECT group_concat(run_id || ':' || status, '|') AS s "
        "FROM (SELECT run_id, status FROM runs ORDER BY started_at DESC LIMIT 60)"
    ).fetchone()["s"] or ""
    cps_sig = conn.execute(
        "SELECT group_concat(checkpoint_id || ':' || status, '|') AS s "
        "FROM (SELECT checkpoint_id, status FROM checkpoints ORDER BY created_at DESC LIMIT 60)"
    ).fetchone()["s"] or ""
    spent = monthly_spent(conn)
    return {
        "max_rowid": max_rowid,
        "runs_sig": runs_sig,
        "cps_sig": cps_sig,
        "spent": round(spent, 4),
    }


def runs_touched_since(conn: sqlite3.Connection, rowid: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT run_id FROM messages WHERE rowid > ?", (rowid,)
    ).fetchall()
    return [r["run_id"] for r in rows]
