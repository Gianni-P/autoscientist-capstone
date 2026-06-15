"""Manual pause / resume state for a run.

The five HITL checkpoints (see KICKOFF.md §7) cover *automatic* gates the
runner stops at. This module covers the *operator-initiated* break: the
Pause button in the Streamlit console. State lives in the ``run_controls``
table (one row per run) and the runner polls it between agents.

Lifecycle
~~~~~~~~~

1. Operator clicks **Pause** → :func:`request_pause` sets
   ``pause_requested=1``. The run keeps running.
2. The runner finishes the current agent (including its full tool loop),
   then calls :func:`is_pause_requested`. If true:
       a. :func:`save_pause_state` records ``next_agent``, ``next_payload``,
          ``handoffs_so_far``, ``code_review_cycles``.
       b. Run status flips to ``paused`` with note ``manual_pause``.
       c. ``_drive_loop`` exits cleanly.
3. Operator clicks **Resume** → :func:`resume_run` reads the saved state,
   :func:`clear_pause_state` wipes the row, and the runner re-enters
   ``_drive_loop`` with the saved counters/payload.

Manual pause and checkpoints are mutually exclusive resume paths but can
coexist briefly: if both a pending checkpoint AND a manual-pause row
exist on the same run, the checkpoint wins (it was opened by the
runner; the manual-pause row is a stale "I clicked pause but the
checkpoint fired first" artifact). :func:`resume_run` clears the stale
row when picking the checkpoint path.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from autoscientist.state.db import now_iso


@dataclass(frozen=True)
class PauseState:
    run_id: str
    pause_requested: bool
    requested_at: str | None
    paused_at: str | None
    next_agent: str | None
    next_payload: str | None
    handoffs_so_far: int | None
    code_review_cycles: int | None

    @property
    def is_active(self) -> bool:
        """True if a manual pause has been honoured (state is saved)."""
        return self.paused_at is not None and self.next_agent is not None


def _row_to_state(row: sqlite3.Row | None) -> PauseState | None:
    if row is None:
        return None
    return PauseState(
        run_id=row["run_id"],
        pause_requested=bool(row["pause_requested"]),
        requested_at=row["requested_at"],
        paused_at=row["paused_at"],
        next_agent=row["next_agent"],
        next_payload=row["next_payload"],
        handoffs_so_far=row["handoffs_so_far"],
        code_review_cycles=row["code_review_cycles"],
    )


def request_pause(conn: sqlite3.Connection, run_id: str) -> None:
    """Operator-side: set the pause_requested flag.

    UPSERT-style — overwrites any prior request_pause without disturbing
    a previously-saved ``next_agent`` / ``next_payload`` if one exists.
    Idempotent: a second click before the runner honours the first is a
    no-op (the flag is already 1).
    """
    conn.execute(
        "INSERT INTO run_controls (run_id, pause_requested, requested_at) "
        "VALUES (?, 1, ?) "
        "ON CONFLICT(run_id) DO UPDATE SET "
        "    pause_requested = 1, "
        "    requested_at = COALESCE(run_controls.requested_at, excluded.requested_at)",
        (run_id, now_iso()),
    )


def cancel_pause_request(conn: sqlite3.Connection, run_id: str) -> None:
    """Operator-side: clear an outstanding pause request that hasn't yet
    been honoured. No-op if the run is already paused (saved state exists)."""
    conn.execute(
        "UPDATE run_controls SET pause_requested = 0, requested_at = NULL "
        "WHERE run_id = ? AND paused_at IS NULL",
        (run_id,),
    )


def is_pause_requested(conn: sqlite3.Connection, run_id: str) -> bool:
    """Runner-side: poll between agents. Returns True if Pause was clicked."""
    row = conn.execute(
        "SELECT pause_requested FROM run_controls WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return bool(row and row["pause_requested"])


def save_pause_state(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    next_agent: str,
    next_payload: str,
    handoffs_so_far: int,
    code_review_cycles: int,
) -> None:
    """Runner-side: record where the loop stopped.

    Called by ``_drive_loop`` once it has decided to honour the pause.
    Stores the next agent and its payload so :func:`read_pause_state` +
    resume can rebuild the loop from where it left off, complete with
    the per-loop counters that cap the revision loop.
    """
    conn.execute(
        "INSERT INTO run_controls ("
        "    run_id, pause_requested, requested_at, paused_at, "
        "    next_agent, next_payload, handoffs_so_far, code_review_cycles"
        ") VALUES (?, 0, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(run_id) DO UPDATE SET "
        "    pause_requested = 0, "
        "    paused_at = excluded.paused_at, "
        "    next_agent = excluded.next_agent, "
        "    next_payload = excluded.next_payload, "
        "    handoffs_so_far = excluded.handoffs_so_far, "
        "    code_review_cycles = excluded.code_review_cycles",
        (run_id, now_iso(), now_iso(), next_agent, next_payload,
         handoffs_so_far, code_review_cycles),
    )


def read_pause_state(conn: sqlite3.Connection, run_id: str) -> PauseState | None:
    """Read the row. Returns ``None`` if no manual pause is recorded."""
    row = conn.execute(
        "SELECT run_id, pause_requested, requested_at, paused_at, "
        "next_agent, next_payload, handoffs_so_far, code_review_cycles "
        "FROM run_controls WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    return _row_to_state(row)


def clear_pause_state(conn: sqlite3.Connection, run_id: str) -> None:
    """Delete the row. Called by resume_run after reading saved state."""
    conn.execute("DELETE FROM run_controls WHERE run_id = ?", (run_id,))
