"""Starlette app for the snappy operator console.

Routes
------
GET  /                                  single-page app shell
GET  /static/*                          css / js assets
GET  /api/overview                      budget + runs + pending + active run
GET  /api/runs                          run list (with counts / cost / agent)
GET  /api/runs/{run_id}                 run detail (stages, totals, pause)
GET  /api/runs/{run_id}/messages        live feed slice (?after=<rowid>)
GET  /api/runs/{run_id}/timeline        per-agent activations + handoffs
GET  /api/checkpoints/{cp_id}           full checkpoint payload + Q&A
GET  /api/messages/{message_id}         one message, full (uncapped) content
GET  /api/agents/{name}/prompt          static system prompt (prompts/<name>.md)
POST /api/checkpoints/{cp_id}/resolve   approve|modify|reject (+resume)
POST /api/checkpoints/{cp_id}/questions ask the orchestrator (persists)
POST /api/runs/{run_id}/pause           request a manual pause
POST /api/runs/{run_id}/cancel-pause    cancel an outstanding pause request
POST /api/runs/{run_id}/resume          resume a manually-paused run
GET  /api/stream                        SSE: pushes change events (no polling)

Read handlers are plain ``def`` (Starlette runs them in a threadpool, so the
blocking sqlite reads never stall the event loop). Mutating handlers are
``async`` and offload their DB work via ``run_in_threadpool``. The SSE loop
tails the DB on a fast interval and pushes only deltas.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import anyio
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.runtime import control as run_control
from autoscientist.runtime.config import load_config
from autoscientist.web import queries
from autoscientist.web.resume import resume_in_background, start_in_background

_DEFAULT_START_AGENT = "lit_review"

_STATIC = Path(__file__).parent / "static"

# How often the SSE loop tails the DB for changes. Sub-half-second → the
# browser is pushed within ~one tick of an event landing, which reads as
# "instant" without hammering the 220 MB DB (the signature query is tiny).
_POLL_INTERVAL = 0.4


# ---------------------------------------------------------------------------
# A schema-ensure-free read connection for the hot SSE loop.
#
# queries.connect() == open_db(), which re-runs CREATE TABLE IF NOT EXISTS +
# a commit on every open. That's fine once per REST request, but the SSE loop
# polls every 0.4 s — doing schema-ensure that often is wasteful and churns
# the WAL against the live runner. This bypasses it (the schema already
# exists by the time the console runs). check_same_thread=False because
# successive run_in_threadpool ticks may land on different pool threads
# (always serialized — we await each).
# ---------------------------------------------------------------------------

def _read_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(load_config().db_path()), timeout=30.0, check_same_thread=False
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def _json(data: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status)


# ---------------------------------------------------------------------------
# Static shell
# ---------------------------------------------------------------------------

async def index(_: Request) -> FileResponse:
    return FileResponse(_STATIC / "index.html")


# ---------------------------------------------------------------------------
# Read endpoints (sync → threadpool)
# ---------------------------------------------------------------------------

def api_overview(_: Request) -> JSONResponse:
    with closing(queries.connect()) as conn:
        return _json(queries.overview(conn))


def api_runs(_: Request) -> JSONResponse:
    with closing(queries.connect()) as conn:
        return _json({"runs": queries.list_runs(conn, limit=200)})


def api_run_detail(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    with closing(queries.connect()) as conn:
        detail = queries.run_detail(conn, run_id)
    if detail is None:
        return _json({"error": "run not found"}, 404)
    return _json(detail)


def api_run_messages(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    try:
        after = int(request.query_params.get("after", "0"))
    except ValueError:
        after = 0
    try:
        limit = min(800, max(1, int(request.query_params.get("limit", "400"))))
    except ValueError:
        limit = 400
    with closing(queries.connect()) as conn:
        return _json(queries.run_messages(conn, run_id, after_rowid=after, limit=limit))


def api_run_timeline(request: Request) -> JSONResponse:
    run_id = request.path_params["run_id"]
    with closing(queries.connect()) as conn:
        return _json({"activations": queries.run_timeline(conn, run_id)})


def api_checkpoint(request: Request) -> JSONResponse:
    cp_id = request.path_params["cp_id"]
    with closing(queries.connect()) as conn:
        detail = queries.checkpoint_detail(conn, cp_id)
    if detail is None:
        return _json({"error": "checkpoint not found"}, 404)
    return _json(detail)


def api_message(request: Request) -> JSONResponse:
    message_id = request.path_params["message_id"]
    with closing(queries.connect()) as conn:
        ev = queries.message_detail(conn, message_id)
    if ev is None:
        return _json({"error": "message not found"}, 404)
    return _json(ev)


def api_agent_prompt(request: Request) -> JSONResponse:
    name = request.path_params["name"]
    data = queries.agent_system_prompt(name)
    if data is None:
        return _json({"error": "no prompt for that agent"}, 404)
    return _json(data)


def api_projects(_: Request) -> JSONResponse:
    return _json({"projects": queries.list_projects(), "default_agent": _DEFAULT_START_AGENT})


def api_project_payload(request: Request) -> JSONResponse:
    payload = queries.project_payload(request.path_params["name"])
    if payload is None:
        return _json({"error": "no kickoff payload for that project"}, 404)
    return _json({"project": request.path_params["name"], "payload": payload})


def api_run_checkpoints(request: Request) -> JSONResponse:
    with closing(queries.connect()) as conn:
        return _json({"checkpoints": queries.run_checkpoints(conn, request.path_params["run_id"])})


# ---------------------------------------------------------------------------
# Action endpoints (async → offload DB work)
# ---------------------------------------------------------------------------

def _resolve_blocking(cp_id: str, decision: str, instructions: str | None,
                      modified_payload: str | None) -> dict[str, Any]:
    with closing(queries.connect()) as conn:
        cp = cp_manager.get_checkpoint(conn, cp_id)
        if cp is None:
            return {"ok": False, "error": "checkpoint not found", "status": 404}
        try:
            rec = cp_manager.resolve(
                conn,
                checkpoint_id=cp_id,
                decision=decision,
                instructions=instructions or None,
                modified_payload=modified_payload or None,
            )
            conn.commit()
        except (ValueError, RuntimeError) as e:
            return {"ok": False, "error": str(e), "status": 409}
        run_id = rec.run_id
    resumed = False
    if decision != cp_manager.DECISION_REJECT:
        resumed = resume_in_background(run_id)
    return {"ok": True, "status": rec.status, "run_id": run_id, "resumed": resumed}


async def api_resolve(request: Request) -> JSONResponse:
    cp_id = request.path_params["cp_id"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    decision = (body or {}).get("decision")
    if decision not in cp_manager._VALID_DECISIONS:
        return _json({"ok": False, "error": "invalid decision"}, 400)
    result = await run_in_threadpool(
        _resolve_blocking,
        cp_id,
        decision,
        (body or {}).get("instructions"),
        (body or {}).get("modified_payload"),
    )
    return _json(result, result.pop("status", 200) if not result.get("ok") else 200)


def _ask_blocking(cp_id: str, content: str) -> dict[str, Any]:
    with closing(queries.connect()) as conn:
        if cp_manager.get_checkpoint(conn, cp_id) is None:
            return {"ok": False, "error": "checkpoint not found", "status": 404}
        qid = cp_manager.add_question(
            conn, checkpoint_id=cp_id, role="operator", content=content
        )
        conn.commit()
    return {"ok": True, "question_id": qid}


async def api_ask(request: Request) -> JSONResponse:
    cp_id = request.path_params["cp_id"]
    try:
        body = await request.json()
    except Exception:
        body = {}
    content = ((body or {}).get("content") or "").strip()
    if not content:
        return _json({"ok": False, "error": "empty question"}, 400)
    result = await run_in_threadpool(_ask_blocking, cp_id, content)
    return _json(result, result.pop("status", 200) if not result.get("ok") else 200)


def _start_blocking(project: str, agent: str | None, payload: str | None) -> dict[str, Any]:
    if not queries.project_exists(project):
        return {"ok": False, "error": f"unknown project: {project}", "status": 404}
    # Default the payload to the project's kickoff_payload.json when not given.
    if not payload:
        payload = queries.project_payload(project) or ""
    started = start_in_background(
        project=project, agent=(agent or _DEFAULT_START_AGENT), payload=payload
    )
    return {"ok": started, "project": project}


async def api_start_run(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    project = (body or {}).get("project")
    if not project:
        return _json({"ok": False, "error": "project is required"}, 400)
    result = await run_in_threadpool(
        _start_blocking, project, (body or {}).get("agent"), (body or {}).get("payload")
    )
    return _json(result, result.pop("status", 200) if not result.get("ok") else 200)


def _pause_blocking(run_id: str, action: str) -> dict[str, Any]:
    if action == "resume":
        return {"ok": True, "resumed": resume_in_background(run_id)}
    with closing(queries.connect()) as conn:
        if action == "pause":
            run_control.request_pause(conn, run_id)
        elif action == "cancel-pause":
            run_control.cancel_pause_request(conn, run_id)
        conn.commit()
    return {"ok": True}


async def api_pause(request: Request) -> JSONResponse:
    return _json(await run_in_threadpool(
        _pause_blocking, request.path_params["run_id"], "pause"))


async def api_cancel_pause(request: Request) -> JSONResponse:
    return _json(await run_in_threadpool(
        _pause_blocking, request.path_params["run_id"], "cancel-pause"))


async def api_manual_resume(request: Request) -> JSONResponse:
    return _json(await run_in_threadpool(
        _pause_blocking, request.path_params["run_id"], "resume"))


# ---------------------------------------------------------------------------
# SSE — push deltas, no client polling
# ---------------------------------------------------------------------------

async def api_stream(request: Request) -> EventSourceResponse:
    async def event_publisher():
        conn = await run_in_threadpool(_read_connection)
        try:
            last = await run_in_threadpool(queries.change_signature, conn)
            # Tell the client the current high-water mark so its feed cursor
            # lines up with what the snapshot already contains.
            yield {"event": "ready", "data": json.dumps({"max_rowid": last["max_rowid"]})}
            while True:
                if await request.is_disconnected():
                    break
                await anyio.sleep(_POLL_INTERVAL)
                try:
                    sig = await run_in_threadpool(queries.change_signature, conn)
                except Exception:
                    continue
                changed: dict[str, Any] = {}
                if sig["max_rowid"] != last["max_rowid"]:
                    runs = await run_in_threadpool(
                        queries.runs_touched_since, conn, last["max_rowid"]
                    )
                    changed["messages"] = {"runs": runs, "max_rowid": sig["max_rowid"]}
                if sig["runs_sig"] != last["runs_sig"]:
                    changed["runs"] = True
                if sig["cps_sig"] != last["cps_sig"]:
                    changed["checkpoints"] = True
                if sig["spent"] != last["spent"]:
                    changed["budget"] = sig["spent"]
                if changed:
                    yield {"event": "update", "data": json.dumps(changed)}
                last = sig
        finally:
            conn.close()

    return EventSourceResponse(event_publisher(), ping=15)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def _health(_: Request) -> Response:
    return JSONResponse({"ok": True})


routes = [
    Route("/", index),
    Route("/healthz", _health),
    Route("/api/overview", api_overview),
    Route("/api/projects", api_projects),
    Route("/api/projects/{name}/payload", api_project_payload),
    # Literal /api/runs/start must precede the /api/runs/{run_id} pattern.
    Route("/api/runs/start", api_start_run, methods=["POST"]),
    Route("/api/runs", api_runs),
    Route("/api/runs/{run_id}", api_run_detail),
    Route("/api/runs/{run_id}/messages", api_run_messages),
    Route("/api/runs/{run_id}/timeline", api_run_timeline),
    Route("/api/runs/{run_id}/checkpoints", api_run_checkpoints),
    Route("/api/runs/{run_id}/pause", api_pause, methods=["POST"]),
    Route("/api/runs/{run_id}/cancel-pause", api_cancel_pause, methods=["POST"]),
    Route("/api/runs/{run_id}/resume", api_manual_resume, methods=["POST"]),
    Route("/api/checkpoints/{cp_id}", api_checkpoint),
    Route("/api/checkpoints/{cp_id}/resolve", api_resolve, methods=["POST"]),
    Route("/api/checkpoints/{cp_id}/questions", api_ask, methods=["POST"]),
    Route("/api/messages/{message_id}", api_message),
    Route("/api/agents/{name}/prompt", api_agent_prompt),
    Route("/api/stream", api_stream),
    Mount("/static", app=StaticFiles(directory=str(_STATIC)), name="static"),
]

app = Starlette(routes=routes)


def main() -> None:
    """Entry point for ``python -m autoscientist.web`` / console script."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="autoscientist-web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8650)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
