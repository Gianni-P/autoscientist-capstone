"""Streamlit operator console + five-stage checkpoint resolver.

Run with::

    uv run streamlit run src/autoscientist/checkpoints/ui.py

The app has two modes, switched by a ``cp`` query parameter:

  * **console** (default): runs table, monthly spend, pending checkpoints,
    live activity stream (every assistant turn / tool call / handoff as
    it lands). Auto-refreshes via ``st.fragment(run_every=...)``.
  * **resolver** (``?cp=<checkpoint_id>``): stage-specific layout for the
    selected checkpoint, with operator actions (approve / reject /
    modify / ask-questions), a Q&A thread, and the activity panel for
    the run leading up to the pause.

KICKOFF.md §7 is the spec — the five stage layouts (idea selection,
methodology approval, preliminary review, full results validation,
draft review) each render the payload the corresponding agent emits.
Layouts fall back to a raw-JSON view when the agent's output didn't
parse, so the UI never blocks the operator on schema drift.

Auto-refresh design
~~~~~~~~~~~~~~~~~~~

The page is broken into ``@st.fragment(run_every=...)`` chunks: each
fragment re-runs on its own timer without rerunning the full page, so
operator forms, expanders, and scroll position are preserved between
ticks. The sidebar has a refresh-interval picker (1s/2s/3s/5s/10s/off)
backed by ``st.session_state``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from typing import Any

import streamlit as st

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.runtime import control as run_control
from autoscientist.runtime.budget import BudgetConfig, monthly_spent
from autoscientist.runtime.config import load_config
from autoscientist.state.db import month_key as get_month_key
from autoscientist.state.db import open_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Per-fragment refresh cadence. Streamlit fixes ``run_every`` at decoration
# time (the decorator runs once on import), so we can't drive intervals from
# session state. Picking sensible static defaults per section instead.
REFRESH_PENDING_CP = "2s"   # high-signal — operator is waiting on this
REFRESH_ACTIVITY   = "3s"   # high-signal — terminal-tail vibe
REFRESH_RUNS_TABLE = "10s"  # low-signal — runs table rarely changes


def _connect() -> sqlite3.Connection:
    cfg = load_config()
    db_path = cfg.db_path()
    return open_db(db_path)


def _selected_cp_id() -> str | None:
    raw = st.query_params.get("cp")
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw or None


def _go_to_console() -> None:
    st.query_params.clear()
    st.rerun()


def _go_to_checkpoint(cp_id: str) -> None:
    st.query_params["cp"] = cp_id
    st.rerun()


def _resume_command(run_id: str) -> list[str]:
    """Argv for resuming a run as its own process (`runner --resume <id>`)."""
    return [sys.executable, "-m", "autoscientist.runtime.runner", "--resume", run_id]


def _resume_in_background(run_id: str) -> None:
    """Launch a resume as a DETACHED subprocess so it outlives the UI.

    Doing the resume inline would freeze the Streamlit page until every
    downstream agent completes; the previous implementation used a daemon
    thread, but the OS kills a daemon thread when the Streamlit process exits
    or restarts — leaving the run wedged in 'running' (un-resumable). The runner
    already runs standalone via ``--resume <run_id>`` (the documented two-process
    model), so we spawn that in a new session: it keeps driving the chain, writes
    its own ``runs/<run_id>/logs/run.jsonl``, and updates run status in the DB
    that the UI polls — independent of this process's lifecycle. A duplicate
    launch is harmless: the second resume_run sees status != 'paused' and exits.
    """
    from autoscientist.runtime.config import load_config

    cmd = _resume_command(run_id)
    popen_kwargs: dict[str, Any] = {
        "cwd": str(load_config().root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "posix":
        # New session so the child isn't in Streamlit's process group and
        # survives the server going away.
        popen_kwargs["start_new_session"] = True
    try:
        subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        # Never crash the page if the launch fails; surface it in the logs.
        import structlog
        structlog.get_logger("autoscientist.ui").exception(
            "ui.resume_launch_failed", run_id=run_id, error=str(e)
        )


# ---------------------------------------------------------------------------
# Live activity stream
# ---------------------------------------------------------------------------

_ROLE_ICONS = {
    "user":      "📨",
    "assistant": "🤖",
    "tool":      "🔧",
    "handoff":   "↪",
    "system":    "⚙",
}

_PREVIEW_CHARS = 280


def _shorten(text: str, limit: int = _PREVIEW_CHARS) -> str:
    if not text:
        return "(empty)"
    flat = " ".join(text.split())  # collapse whitespace for the one-liner preview
    return flat if len(flat) <= limit else flat[:limit] + " …"


def _latest_active_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """Most-recent run in 'running' or 'paused' state, else the most recent overall."""
    row = conn.execute(
        "SELECT run_id, project_id, status, started_at, ended_at, note "
        "FROM runs WHERE status IN ('running','paused') "
        "ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return row
    return conn.execute(
        "SELECT run_id, project_id, status, started_at, ended_at, note "
        "FROM runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()


def _recent_messages(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    limit: int = 60,
) -> list[sqlite3.Row]:
    """Last ``limit`` messages, most-recent-first, optionally for one run."""
    if run_id:
        return list(conn.execute(
            "SELECT message_id, run_id, agent_name, role, content, model, "
            "prompt_tokens, completion_tokens, cost_usd, latency_ms, cache_hit, "
            "created_at "
            "FROM messages WHERE run_id = ? ORDER BY created_at DESC LIMIT ?",
            (run_id, limit),
        ))
    return list(conn.execute(
        "SELECT message_id, run_id, agent_name, role, content, model, "
        "prompt_tokens, completion_tokens, cost_usd, latency_ms, cache_hit, "
        "created_at "
        "FROM messages ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ))


def _render_message_card(row: sqlite3.Row) -> None:
    """One event row → one Streamlit container, terminal-log style.

    Each event is a single line with an icon, an agent badge, the
    role-specific summary (model+tokens+cost for assistants, tool name
    and timing for tool calls), and a content preview. The full content
    sits behind an expander so the stream stays scannable.
    """
    role = row["role"]
    agent = row["agent_name"]
    icon = _ROLE_ICONS.get(role, "·")
    ts = row["created_at"]
    # Drop the date portion if present — keep just HH:MM:SS.fff for the line.
    ts_short = ts.split("T", 1)[1] if "T" in ts else ts
    ts_short = ts_short.split(".", 1)[0]

    if role == "assistant":
        model = row["model"] or "?"
        pt = row["prompt_tokens"] or 0
        ct = row["completion_tokens"] or 0
        cost = row["cost_usd"] or 0.0
        latency = row["latency_ms"] or 0
        cache = " · cached" if row["cache_hit"] else ""
        header = (
            f"`{ts_short}` {icon} **{agent}** · {model} · "
            f"{pt}→{ct} tok · ${float(cost):.4f} · {latency}ms{cache}"
        )
        content = row["content"] or ""
        st.markdown(header)
        st.caption(_shorten(content))
        if content and len(content) > _PREVIEW_CHARS:
            with st.expander("full assistant content", expanded=False):
                st.code(content, language="text")
        return

    if role == "tool":
        # content is a JSON blob {tool_use_id, name, input, output, error, duration_ms}
        raw = row["content"] or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        name = data.get("name", "?")
        duration = data.get("duration_ms", row["latency_ms"] or 0)
        error = data.get("error")
        inp = data.get("input") or {}
        outp = data.get("output")
        # One-liner summarising the input keys/values.
        input_preview = _shorten(
            ", ".join(f"{k}={json.dumps(v) if not isinstance(v, str) else v!r}"
                      for k, v in (inp.items() if isinstance(inp, dict) else []))
            or json.dumps(inp)[:200],
            limit=180,
        )
        status_icon = "❌" if error else "✓"
        st.markdown(
            f"`{ts_short}` {icon} **{agent}** · `{name}` · {duration}ms {status_icon}"
        )
        st.caption(input_preview)
        if error or (isinstance(outp, (dict, list, str)) and outp):
            with st.expander(f"tool full I/O — {name}", expanded=False):
                if error:
                    st.error(f"error: {error}")
                st.markdown("**input**")
                st.json(inp, expanded=False)
                if outp is not None:
                    st.markdown("**output**")
                    if isinstance(outp, (dict, list)):
                        st.json(outp, expanded=False)
                    else:
                        st.code(str(outp), language="text")
        return

    if role == "user":
        content = row["content"] or ""
        st.markdown(f"`{ts_short}` {icon} **inbound → {agent}**")
        st.caption(_shorten(content))
        if content and len(content) > _PREVIEW_CHARS:
            with st.expander(f"full inbound payload to {agent}", expanded=False):
                st.code(content, language="text")
        return

    if role == "handoff":
        content = row["content"] or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}
        cp_id = data.get("checkpoint_id", "?")
        decision = data.get("decision", "?")
        next_agent = data.get("next_agent", "?")
        st.markdown(
            f"`{ts_short}` {icon} **resume** · cp=`{cp_id}` · "
            f"decision=`{decision}` · → `{next_agent}`"
        )
        return

    # Fallback: system or unknown roles.
    st.markdown(f"`{ts_short}` {icon} **{agent}** · `{role}`")
    st.caption(_shorten(row["content"] or ""))


def _render_activity_stream(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
    limit: int = 60,
) -> None:
    """Render the last N messages as a terminal-log-style stream.

    When ``run_id`` is None, shows the latest activity across all runs;
    when set, scopes to that run. Newest at the top.
    """
    rows = _recent_messages(conn, run_id=run_id, limit=limit)
    if not rows:
        st.info("No activity yet. The stream populates as agents emit messages, "
                "tool calls land, and handoffs fire.")
        return
    # Subtle separators between rows; rendering top-down (most recent first).
    for i, row in enumerate(rows):
        _render_message_card(row)
        if i < len(rows) - 1:
            st.divider()


# ---------------------------------------------------------------------------
# Console view
# ---------------------------------------------------------------------------

def render_console(conn: sqlite3.Connection) -> None:
    """Console page: budget, pending checkpoints, live activity, recent runs.

    Layout:
      * sidebar refresh-interval picker (1s/2s/3s/5s/10s/off, default 3s)
      * top-of-page budget metrics (recomputed on every page run)
      * three auto-refreshing fragments:
          - pending checkpoints (high priority — refresh fast)
          - live activity stream (the headline new feature)
          - recent runs table (low priority — slow refresh acceptable)

    Each fragment opens its own sqlite connection so concurrent
    refreshes don't trip on cursor state. The caller's connection
    handles the static header.
    """
    cfg = load_config()
    bcfg = BudgetConfig.from_dict(cfg.models.get("budget", {}))
    spent = monthly_spent(conn)

    col1, col2, col3 = st.columns(3)
    col1.metric("Month", get_month_key())
    col2.metric("Real spend", f"${spent:.2f}")
    col3.metric("Cap", f"${bcfg.monthly_cap_usd:.2f}")
    st.progress(min(1.0, spent / bcfg.monthly_cap_usd))

    # The three live sections. Fragments re-run on their own timer; forms
    # outside (none on this page) and other widgets stay intact.
    _pending_checkpoints_fragment()
    _activity_stream_fragment()
    _recent_runs_fragment()


@st.fragment(run_every=REFRESH_PENDING_CP)
def _pending_checkpoints_fragment() -> None:
    st.subheader("Pending checkpoints")
    with closing(_connect()) as conn:
        pending = cp_manager.list_pending(conn)
    if not pending:
        st.success(
            "No pending checkpoints. Runs continue automatically when no "
            "operator action is required."
        )
        return
    for cp in pending:
        title = cp_manager.STAGE_TITLES.get(cp.stage, f"Stage {cp.stage}")
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(
                f"**{title}**  \n`run_id={cp.run_id}`  ·  opened `{cp.created_at}`"
            )
            left.caption(
                f"from `{cp.from_agent}` → next `{cp.to_agent or 'DONE'}`"
            )
            # Flag forced loop-cap CP3s right on the list so they stand out.
            extra = cp.extra or {}
            if extra.get("loop_cap_exceeded"):
                left.warning(
                    f"⚠ revision-loop cap hit ({extra.get('cycles', '?')}/"
                    f"{extra.get('max_cycles', '?')})"
                )
            if right.button("Open", key=f"open_{cp.checkpoint_id}"):
                _go_to_checkpoint(cp.checkpoint_id)


@st.fragment(run_every=REFRESH_ACTIVITY)
def _activity_stream_fragment() -> None:
    """Console-level live activity panel.

    Shows the most-recent active (or paused) run's tail of messages, plus
    a small per-run-status header and Pause/Resume controls. The resolver
    view ties activity to a specific checkpoint's run.
    """
    st.subheader("Live activity")
    with closing(_connect()) as conn:
        run = _latest_active_run(conn)
        if run is None:
            st.info(
                "No runs yet. Launch a run with "
                "`python -m autoscientist.runtime.runner --agent lit_review ...`"
            )
            return
        status_color = {
            "running":   "🟢",
            "paused":    "🟡",
            "completed": "✅",
            "failed":    "🔴",
            "cancelled": "⚪",
        }.get(run["status"], "·")
        st.caption(
            f"{status_color} run `{run['run_id']}` · project "
            f"`{run['project_id']}` · status **{run['status']}** · "
            f"started `{run['started_at']}`"
        )
        _render_run_controls(conn, run["run_id"], run["status"])
        _render_activity_stream(conn, run_id=run["run_id"], limit=40)


def _render_run_controls(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
) -> None:
    """Pause / Resume / Cancel-pause-request controls for the active run.

    The button set is context-aware based on the run's status and any
    ``run_controls`` row:

    * ``running`` + no pending pause           → **Pause**
    * ``running`` + pause already requested    → "Pause requested..."
    * ``paused`` + manual-pause state present  → **Resume**
    * ``paused`` + no manual-pause state       → resolve a checkpoint instead
    * ``completed`` / ``failed`` / ``cancelled`` → no controls
    """
    if status not in {"running", "paused"}:
        return

    pause = run_control.read_pause_state(conn, run_id)
    pending_cps = [c for c in cp_manager.list_for_run(conn, run_id)
                   if c.status == "pending"]

    col1, col2, col3 = st.columns([1, 1, 4])

    if status == "running":
        if pause is not None and pause.pause_requested and pause.paused_at is None:
            # Pause was clicked, runner hasn't honoured it yet.
            col1.button("⏸ Pause requested…", disabled=True,
                        key=f"pausing_{run_id}", use_container_width=True)
            if col2.button("Cancel pause", key=f"cancel_pause_{run_id}",
                           use_container_width=True):
                run_control.cancel_pause_request(conn, run_id)
                conn.commit()
                st.rerun(scope="fragment")
            col3.caption(
                "Waiting for the current agent to finish its tool loop "
                f"(requested at `{pause.requested_at}`)."
            )
        else:
            if col1.button("⏸ Pause", key=f"pause_{run_id}",
                           use_container_width=True, type="primary"):
                run_control.request_pause(conn, run_id)
                conn.commit()
                st.rerun(scope="fragment")
            col3.caption(
                "Pauses the chain at the next agent boundary (after the "
                "current agent's tool loop finishes). Use **Resume** from "
                "this panel when you're ready to continue."
            )
        return

    # status == "paused"
    if pending_cps:
        col3.caption(
            f"Run is paused at a checkpoint ({len(pending_cps)} pending). "
            "Open the pending checkpoint above to approve/modify/reject — "
            "that's how runs at HITL gates resume."
        )
        return

    if pause is not None and pause.is_active:
        if col1.button("▶ Resume", key=f"resume_{run_id}",
                       use_container_width=True, type="primary"):
            _resume_in_background(run_id)
            st.toast("Resuming run in the background.", icon="▶")
            st.rerun(scope="fragment")
        col3.caption(
            f"Manual pause from `{pause.paused_at}` · next agent "
            f"`{pause.next_agent}` · {pause.handoffs_so_far or 0} handoffs done."
        )
        return

    col3.caption(
        "Run is paused but has no pending checkpoint and no manual-pause "
        "state — this usually means a KeyboardInterrupt or a hand-edited "
        "row. Inspect the run's logs before continuing."
    )


@st.fragment(run_every=REFRESH_RUNS_TABLE)
def _recent_runs_fragment() -> None:
    st.subheader("Recent runs")
    with closing(_connect()) as conn:
        run_rows = conn.execute(
            "SELECT run_id, project_id, status, started_at, ended_at, note "
            "FROM runs ORDER BY started_at DESC LIMIT 50"
        ).fetchall()
    if not run_rows:
        st.info("No runs yet.")
        return
    st.dataframe(
        [
            {
                "run_id": r["run_id"],
                "project": r["project_id"],
                "status": r["status"],
                "started_at": r["started_at"],
                "ended_at": r["ended_at"],
                "note": r["note"],
            }
            for r in run_rows
        ],
        hide_index=True,
        width="stretch",
    )


# ---------------------------------------------------------------------------
# Stage-specific renderers
# ---------------------------------------------------------------------------

def _render_raw_fallback(parsed: dict[str, Any] | None, raw: str) -> None:
    if parsed is not None:
        st.json(parsed, expanded=False)
    else:
        st.warning("Agent output did not parse as JSON. Showing raw text.")
        st.code(raw or "(empty)", language="text")


def _render_stage1_idea_selection(parsed: dict[str, Any] | None, raw: str) -> None:
    """Idea critic emits: critiques + ranked_indices + top_pick + operator_questions."""
    if parsed is None:
        _render_raw_fallback(parsed, raw)
        return

    top_pick = parsed.get("top_pick")
    ranked = parsed.get("ranked_indices") or []
    critiques = parsed.get("critiques") or []
    operator_qs = parsed.get("operator_questions") or []

    st.markdown(f"**Critic top pick:** idea index `{top_pick}`  \n**Ranked:** `{ranked}`")
    if operator_qs:
        st.info("**Critic asks the operator:**\n\n" + "\n".join(f"- {q}" for q in operator_qs))

    for i, c in enumerate(critiques):
        idx = c.get("idea_index", i)
        rec = (c.get("recommendation") or "").upper() or "?"
        with st.expander(f"Idea {idx} — recommendation: **{rec}**", expanded=(idx == top_pick)):
            for concern in c.get("concerns") or []:
                st.markdown(f"- ⚠ {concern}")
            if c.get("kill_criteria"):
                st.markdown("**Kill criteria:**")
                for kc in c["kill_criteria"]:
                    st.markdown(f"- {kc}")
            if c.get("potential_confounds"):
                st.markdown("**Potential confounds:** " + ", ".join(c["potential_confounds"]))
            if c.get("rationale"):
                st.caption(c["rationale"])


def _render_stage2_methodology(parsed: dict[str, Any] | None, raw: str) -> None:
    if parsed is None:
        _render_raw_fallback(parsed, raw)
        return
    plan = parsed.get("plan") or parsed
    st.markdown(f"**Research question:** {plan.get('research_question', '?')}")

    if plan.get("hypotheses"):
        st.markdown("**Hypotheses:**")
        for h in plan["hypotheses"]:
            st.markdown(
                f"- `{h.get('id', '?')}`: {h.get('statement', '?')} "
                f"(predicted direction: `{h.get('predicted_direction', '?')}`)"
            )

    if plan.get("datasets"):
        st.markdown("**Datasets:**")
        st.dataframe(plan["datasets"], hide_index=True, width="stretch")

    if plan.get("baselines"):
        st.markdown("**Baselines:**")
        st.dataframe(plan["baselines"], hide_index=True, width="stretch")

    if plan.get("metrics"):
        st.markdown("**Metrics:**")
        st.dataframe(plan["metrics"], hide_index=True, width="stretch")

    if plan.get("experiments"):
        st.markdown("**Experiments:**")
        st.dataframe(plan["experiments"], hide_index=True, width="stretch")

    sp = plan.get("stats_plan") or {}
    if sp:
        st.markdown(
            f"**Stats plan:** {sp.get('primary_test', '?')}  ·  α=`{sp.get('alpha', '?')}`  ·  "
            f"MC adj=`{sp.get('multiple_comparisons', '?')}`  ·  "
            f"effect floor=`{sp.get('effect_size_floor', '?')}`"
        )

    pa = plan.get("pitfall_acks") or []
    if pa:
        st.markdown("**Pitfall acknowledgements:**")
        for p in pa:
            st.markdown(f"- `{p.get('pitfall', '?')}` → mitigation: {p.get('mitigation', '?')}")

    sc = plan.get("stop_conditions") or {}
    if sc:
        st.markdown(
            f"**Stop conditions:** early-success → `{sc.get('early_success', '?')}`; "
            f"early-abort → `{sc.get('early_abort', '?')}`"
        )


def _render_stage3_preliminary(parsed: dict[str, Any] | None, raw: str) -> None:
    """Preliminary review — code_review verdict or a results_validator subset.

    Stage 3 fires from two places today:
      * After ``code_review`` when the verdict advances forward to
        ``results_validator`` (preferred — gates the full-results run).
      * Forced from the runner when the code_gen ↔ code_review revision
        loop has hit ``runtime.max_code_review_cycles``. ``extra.loop_cap_exceeded``
        is rendered as a banner by the resolver layout above this view.

    Until results_validator gains a ``run_kind=preliminary`` path, the
    "subset results" framing in KICKOFF §7 stays aspirational; CP3 today
    is the code_review gate.
    """
    if parsed is None:
        _render_raw_fallback(parsed, raw)
        return

    # Detect code_review shape (findings + verdict) vs validator shape (checks).
    is_code_review = ("findings" in parsed or "verdict" in parsed) and "checks" not in parsed
    is_validator = "checks" in parsed or "counterintuitive_findings" in parsed

    if is_code_review and not is_validator:
        _render_code_review_body(parsed)
        return
    if is_validator:
        st.markdown(
            "**Subset run results.** Use these to sanity-check before "
            "committing GPU-hours to the full run."
        )
        _render_validator_body(parsed)
        return
    _render_raw_fallback(parsed, raw)


def _render_code_review_body(parsed: dict[str, Any]) -> None:
    """Render the code_review JSON shape: findings + verdict + summary."""
    verdict = parsed.get("verdict", "?")
    verdict_label = {
        "pass": "✅ pass — code_review wants to advance to results_validator",
        "revise": "🔁 revise — code_review wants another code_gen pass",
        "block": "⛔ block — code_review halted on a blocker finding",
    }.get(verdict, f"verdict: `{verdict}`")
    st.markdown(f"**Code review verdict:** {verdict_label}")
    if parsed.get("summary"):
        st.info(parsed["summary"])

    findings = parsed.get("findings") or []
    if findings:
        rows = [
            {
                "severity": f.get("severity", "?"),
                "category": f.get("category", "?"),
                "file": f.get("file", "?"),
                "lines": f.get("lines", "?"),
                "issue": f.get("issue", ""),
                "fix": f.get("fix_suggestion", ""),
            }
            for f in findings
            if isinstance(f, dict)
        ]
        st.markdown("**Findings:**")
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.success("No findings recorded.")


def _render_stage4_full_results(parsed: dict[str, Any] | None, raw: str) -> None:
    """Full results validation — same schema as preliminary, full data."""
    if parsed is None:
        _render_raw_fallback(parsed, raw)
        return
    st.markdown("**Full results validation.** Counterintuitive findings or anomalies must be addressed before drafting.")
    _render_validator_body(parsed)


def _render_validator_body(parsed: dict[str, Any]) -> None:
    verdict = parsed.get("verdict", "?")
    st.markdown(f"**Validator verdict:** `{verdict}`")
    if parsed.get("operator_payload"):
        st.info(parsed["operator_payload"])

    checks = parsed.get("checks") or []
    if checks:
        rows = [
            {
                "check": c.get("name", "?"),
                "status": c.get("status", "?"),
                "detail": c.get("detail", ""),
            }
            for c in checks
        ]
        st.markdown("**Checks:**")
        st.dataframe(rows, hide_index=True, width="stretch")

    cf = parsed.get("counterintuitive_findings") or []
    if cf:
        st.warning(
            "**Counterintuitive findings flagged.** "
            "These contradict the methodology agent's predicted direction."
        )
        for f in cf:
            st.markdown(f"- {f if isinstance(f, str) else json.dumps(f)}")
    anomalies = parsed.get("anomalies") or []
    if anomalies:
        st.warning("**Anomalies:**")
        for a in anomalies:
            st.markdown(f"- {a if isinstance(a, str) else json.dumps(a)}")


def _render_stage5_draft_review(parsed: dict[str, Any] | None, raw: str) -> None:
    if parsed is None:
        _render_raw_fallback(parsed, raw)
        return
    review = parsed.get("review") or parsed
    rec = parsed.get("recommendation", review.get("recommendation", "?"))
    score = parsed.get("score", review.get("score", "?"))
    st.markdown(f"**Reviewer recommendation:** `{rec}`  ·  **score:** `{score}`")
    if review.get("summary"):
        st.markdown("**Summary:** " + review["summary"])

    strengths = review.get("strengths") or []
    if strengths:
        st.markdown("**Strengths:**")
        for s in strengths:
            st.markdown(f"- {s}")

    weaknesses = review.get("weaknesses") or []
    if weaknesses:
        st.markdown("**Weaknesses:**")
        for w in weaknesses:
            sev = w.get("severity", "?") if isinstance(w, dict) else "?"
            issue = w.get("issue", "?") if isinstance(w, dict) else str(w)
            fix = w.get("suggested_fix", "") if isinstance(w, dict) else ""
            st.markdown(f"- **[{sev}]** {issue}  \n  ↳ fix: {fix}")

    rc = review.get("requested_changes") or []
    if rc:
        st.markdown("**Requested changes:**")
        for c in rc:
            st.markdown(f"- {c}")

    mp = review.get("missed_pitfalls") or []
    if mp:
        st.warning("**Pitfalls the reviewer thinks autoscientist missed:**")
        for p in mp:
            st.markdown(f"- {p}")


_STAGE_RENDERERS = {
    1: _render_stage1_idea_selection,
    2: _render_stage2_methodology,
    3: _render_stage3_preliminary,
    4: _render_stage4_full_results,
    5: _render_stage5_draft_review,
}


# ---------------------------------------------------------------------------
# Resolver view
# ---------------------------------------------------------------------------

def render_resolver(conn: sqlite3.Connection, cp_id: str) -> None:
    cp = cp_manager.get_checkpoint(conn, cp_id)
    if cp is None:
        st.error(f"No checkpoint with id `{cp_id}`.")
        if st.button("← Back to console"):
            _go_to_console()
        return

    title = cp_manager.STAGE_TITLES.get(cp.stage, f"Stage {cp.stage}")
    st.markdown(f"## {title}")
    st.caption(
        f"`checkpoint_id={cp.checkpoint_id}`  ·  `run_id={cp.run_id}`  ·  "
        f"opened `{cp.created_at}`  ·  status: **{cp.status}**"
    )
    st.markdown(
        f"`from {cp.from_agent}` → next `{cp.to_agent or 'DONE'}`"
    )

    if st.button("← Back to console"):
        _go_to_console()
        return

    # Runtime metadata banner (e.g., the code_review loop-cap forced CP3).
    extra = cp.extra or {}
    if extra.get("loop_cap_exceeded"):
        cycles = extra.get("cycles", "?")
        max_cycles = extra.get("max_cycles", "?")
        st.error(
            f"**Revision loop cap exceeded.** `code_review` has fired "
            f"`{cycles}` times in this run, hitting the configured cap of "
            f"`{max_cycles}` (`runtime.max_code_review_cycles`). The next "
            f"step is back to `code_gen`. Decide whether to **approve** "
            f"(let code_gen retry once more), **modify** (give code_gen "
            f"explicit instructions to break the loop), or **reject** "
            f"(cancel the run)."
        )

    renderer = _STAGE_RENDERERS.get(cp.stage, _render_raw_fallback)
    if cp.stage in _STAGE_RENDERERS:
        renderer(cp.parsed, cp.payload.get("agent_output_raw", ""))  # type: ignore[arg-type]
    else:
        _render_raw_fallback(cp.parsed, cp.payload.get("agent_output_raw", ""))

    with st.expander("Raw agent output", expanded=False):
        st.code(cp.payload.get("agent_output_raw", ""), language="text")
    with st.expander("Default next-agent payload", expanded=False):
        st.code(cp.default_payload or "(empty)", language="text")

    # ---------------- operator actions ----------------
    if cp.status == "pending":
        st.subheader("Decide")
        with st.form(key=f"resolve_{cp.checkpoint_id}"):
            decision = st.radio(
                "Decision",
                options=[cp_manager.DECISION_APPROVE, cp_manager.DECISION_MODIFY, cp_manager.DECISION_REJECT],
                horizontal=True,
            )
            instructions = st.text_area(
                "Free-text instructions to the next agent (optional for approve, used by modify)",
                key=f"instr_{cp.checkpoint_id}",
                height=120,
            )
            modified_payload = st.text_area(
                "Override payload (modify only — leave blank to keep the default and append instructions)",
                key=f"mod_{cp.checkpoint_id}",
                height=120,
            )
            submitted = st.form_submit_button("Submit decision and resume run")
            if submitted:
                cp_manager.resolve(
                    conn,
                    checkpoint_id=cp.checkpoint_id,
                    decision=decision,
                    instructions=instructions or None,
                    modified_payload=modified_payload or None,
                )
                conn.commit()
                if decision == cp_manager.DECISION_REJECT:
                    st.error("Rejected. Calling resume to mark the run cancelled.")
                else:
                    st.success("Resolved. Resuming the run in the background.")
                _resume_in_background(cp.run_id)
                _go_to_console()
                return
    else:
        op = cp.operator_input or {}
        st.success(
            f"Resolved as `{cp.status}` at `{cp.resolved_at}` "
            f"(decision: `{op.get('decision', '?')}`)."
        )
        if op.get("instructions"):
            st.markdown("**Operator instructions:**")
            st.code(op["instructions"], language="text")
        if op.get("modified_payload"):
            with st.expander("Operator-overridden payload", expanded=False):
                st.code(op["modified_payload"], language="text")

    # ---------------- Q&A thread ----------------
    st.subheader("Ask the orchestrator")
    qs = cp_manager.list_questions(conn, cp.checkpoint_id)
    if qs:
        for q in qs:
            with st.chat_message("user" if q.role == "operator" else "assistant"):
                st.markdown(q.content)
                meta_bits = [q.created_at]
                if q.agent_used:
                    meta_bits.append(f"via {q.agent_used}")
                if q.cost_usd is not None:
                    meta_bits.append(f"${q.cost_usd:.4f}")
                st.caption(" · ".join(meta_bits))
    with st.form(key=f"ask_{cp.checkpoint_id}", clear_on_submit=True):
        question = st.text_area(
            "Question (persists; does not resolve the checkpoint)",
            key=f"q_{cp.checkpoint_id}",
            height=80,
        )
        ask = st.form_submit_button("Add question")
        if ask and question.strip():
            cp_manager.add_question(
                conn,
                checkpoint_id=cp.checkpoint_id,
                role="operator",
                content=question.strip(),
            )
            conn.commit()
            st.rerun()

    # ---------------- run activity (live tail) ----------------
    st.subheader("Run activity")
    st.caption(
        "Every assistant turn, tool call, inbound payload, and resume "
        "handoff for this run, newest first. Auto-refreshes — useful "
        "when the run resumes through another path while you're here."
    )
    _resolver_activity_fragment(cp.run_id)


@st.fragment(run_every=REFRESH_ACTIVITY)
def _resolver_activity_fragment(run_id: str) -> None:
    """Live-tailing activity panel pinned to a single run.

    Used by the resolver page so the operator sees what the agent
    actually produced (and which tools it called) before deciding
    approve/modify/reject. ``run_id`` is captured at fragment-creation
    time and stays constant across reruns — switching to a different
    checkpoint reloads the resolver and re-creates the fragment.
    """
    with closing(_connect()) as conn:
        _render_activity_stream(conn, run_id=run_id, limit=40)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    """Sidebar with refresh cadence info + force-refresh button.

    The actual refresh is handled by ``st.fragment(run_every=...)``
    decorators; this sidebar just makes the cadence visible so the
    operator knows what to expect, and offers a manual rerun for when
    they want to bypass the timer.
    """
    with st.sidebar:
        st.markdown("**Auto-refresh**")
        st.markdown(
            f"- pending checkpoints: every `{REFRESH_PENDING_CP}`\n"
            f"- live activity stream: every `{REFRESH_ACTIVITY}`\n"
            f"- recent runs table: every `{REFRESH_RUNS_TABLE}`"
        )
        if st.button("Refresh now", use_container_width=True):
            st.rerun()
        st.divider()
        st.caption(
            "Each fragment re-runs on its own timer. Forms, expanders, "
            "and scroll position are preserved between ticks."
        )


def main() -> None:
    st.set_page_config(page_title="autoscientist", layout="wide")
    st.title("autoscientist — operator console")

    cfg = load_config()
    db_path = cfg.db_path()
    if not db_path.exists():
        st.warning(
            f"No DB yet at `{db_path}`. "
            "Run `uv run python scripts/smoke_phase1.py` to create one."
        )
        return

    _render_sidebar()

    conn = open_db(db_path)
    try:
        cp_id = _selected_cp_id()
        if cp_id:
            render_resolver(conn, cp_id)
        else:
            render_console(conn)
    finally:
        conn.close()


def _running_under_streamlit() -> bool:
    """True when executed by ``streamlit run`` (a script-run context exists)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


# Render only when actually launched (streamlit run, or `python ui.py`), NOT on
# a plain import — so the module's helpers can be imported and unit-tested.
if __name__ == "__main__" or _running_under_streamlit():
    main()
