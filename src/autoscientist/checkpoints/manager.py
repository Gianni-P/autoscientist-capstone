"""Human-in-the-loop checkpoint management.

KICKOFF.md §4 #5 (non-negotiable): "No autonomous run-to-completion."
This module is the gate. The runner pauses after specific agents fire;
the operator approves / rejects / modifies / asks questions through the
Streamlit UI; resume picks up from where the chain stopped.

Stage mapping (KICKOFF.md §7):
    1 idea selection         — after ``idea_critic``
    2 methodology approval   — after ``methodology``
    3 preliminary review     — after ``code_review`` when verdict advances
                                forward to results_validator (the
                                "preliminary results" framing in KICKOFF
                                §7 is the eventual subset-run gate; today
                                CP3 is the code_review-forward gate that
                                also caps the code_gen ↔ code_review
                                revision loop — see runner._drive_loop)
    4 full results           — after ``results_validator`` on the full run
    5 draft review           — after ``peer_reviewer``

Stage 3 is conditional: ``stage_for_agent("code_review", handoff_to=...)``
returns the stage only when ``handoff_to != "code_gen"`` (i.e., not a
revise loop), or when the runner explicitly forces it via the loop-cap
path. This keeps a one-shot ``pass`` from invoking the operator twice and
a normal revise cycle from spamming pauses.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from autoscientist.state.db import new_id, now_iso

CHECKPOINT_POLICY: dict[str, tuple[int, str]] = {
    "idea_critic":       (1, "idea_selection"),
    "methodology":       (2, "methodology_approval"),
    "code_review":       (3, "preliminary_review"),
    "results_validator": (4, "full_results_validation"),
    "peer_reviewer":     (5, "draft_review"),
}

STAGE_NAMES: dict[int, str] = {
    1: "idea_selection",
    2: "methodology_approval",
    3: "preliminary_review",
    4: "full_results_validation",
    5: "draft_review",
}

STAGE_TITLES: dict[int, str] = {
    1: "Stage 1 — Idea selection",
    2: "Stage 2 — Methodology approval",
    3: "Stage 3 — Preliminary review",
    4: "Stage 4 — Full results validation",
    5: "Stage 5 — Draft review",
}

DECISION_APPROVE = "approve"
DECISION_REJECT = "reject"
DECISION_MODIFY = "modify"
_VALID_DECISIONS = frozenset({DECISION_APPROVE, DECISION_REJECT, DECISION_MODIFY})

_STATUS_FOR_DECISION = {
    DECISION_APPROVE: "approved",
    DECISION_REJECT: "rejected",
    DECISION_MODIFY: "modified",
}


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    run_id: str
    stage: int
    payload: dict[str, Any]
    status: str  # pending|approved|rejected|modified
    operator_input: dict[str, Any] | None
    created_at: str
    resolved_at: str | None

    @property
    def stage_name(self) -> str:
        return self.payload.get("stage_name") or STAGE_NAMES.get(self.stage, f"stage_{self.stage}")

    @property
    def from_agent(self) -> str:
        return self.payload.get("from_agent", "")

    @property
    def to_agent(self) -> str:
        return self.payload.get("to_agent", "")

    @property
    def default_payload(self) -> str:
        return self.payload.get("default_payload", "")

    @property
    def parsed(self) -> dict[str, Any] | None:
        p = self.payload.get("parsed")
        return p if isinstance(p, dict) else None

    @property
    def extra(self) -> dict[str, Any] | None:
        """Optional runtime metadata attached at checkpoint open time.

        The loop-cap forced CP3 sets ``{"loop_cap_exceeded": True,
        "cycles": N, "max_cycles": M}`` here so the UI can flag it.
        """
        e = self.payload.get("extra")
        return e if isinstance(e, dict) else None


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    checkpoint_id: str
    role: str  # operator|assistant
    content: str
    created_at: str
    agent_used: str | None
    cost_usd: float | None


def stage_for_agent(
    agent_name: str,
    *,
    handoff_to: str | None = None,
) -> tuple[int, str] | None:
    """Return (stage, stage_name) if this agent triggers a checkpoint.

    For ``code_review``, the stage 3 gate opens only when the verdict
    advances the chain forward (handoff target is anything other than
    ``code_gen``). A revise/block handoff back to ``code_gen`` is the
    revision loop and is bounded by the runner's per-run cycle cap
    (``runtime.max_code_review_cycles``), not by a per-iteration
    checkpoint — otherwise the operator would face N pauses for what
    is meant to be an automated revision pass.

    Other agents in the policy ignore ``handoff_to``.
    """
    info = CHECKPOINT_POLICY.get(agent_name)
    if info is None:
        return None
    if agent_name == "code_review" and handoff_to == "code_gen":
        return None
    return info


def open_checkpoint(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    stage: int,
    from_agent: str,
    to_agent: str,
    agent_output_raw: str,
    default_payload: str,
    parsed: dict[str, Any] | None = None,
    summary: str | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Create a pending checkpoint row. Returns checkpoint_id."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown stage: {stage}")
    envelope: dict[str, Any] = {
        "stage": stage,
        "stage_name": STAGE_NAMES[stage],
        "from_agent": from_agent,
        "to_agent": to_agent,
        "agent_output_raw": agent_output_raw,
        "parsed": parsed,
        "default_payload": default_payload,
    }
    if summary:
        envelope["summary"] = summary
    if extra:
        envelope["extra"] = extra
    cp_id = new_id("cp_")
    conn.execute(
        "INSERT INTO checkpoints (checkpoint_id, run_id, stage, payload, status, created_at) "
        "VALUES (?, ?, ?, ?, 'pending', ?)",
        (cp_id, run_id, stage, json.dumps(envelope), now_iso()),
    )
    return cp_id


def get_checkpoint(conn: sqlite3.Connection, checkpoint_id: str) -> CheckpointRecord | None:
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
    ).fetchone()
    return _row_to_record(row) if row else None


def latest_checkpoint(conn: sqlite3.Connection, run_id: str) -> CheckpointRecord | None:
    row = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return _row_to_record(row) if row else None


def list_pending(conn: sqlite3.Connection) -> list[CheckpointRecord]:
    rows = conn.execute(
        "SELECT * FROM checkpoints WHERE status = 'pending' ORDER BY created_at DESC"
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def list_for_run(conn: sqlite3.Connection, run_id: str) -> list[CheckpointRecord]:
    rows = conn.execute(
        "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY created_at ASC",
        (run_id,),
    ).fetchall()
    return [_row_to_record(r) for r in rows]


def _row_to_record(row: sqlite3.Row) -> CheckpointRecord:
    payload = json.loads(row["payload"]) if row["payload"] else {}
    op_input = json.loads(row["operator_input"]) if row["operator_input"] else None
    return CheckpointRecord(
        checkpoint_id=row["checkpoint_id"],
        run_id=row["run_id"],
        stage=row["stage"],
        payload=payload,
        status=row["status"],
        operator_input=op_input,
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )


def resolve(
    conn: sqlite3.Connection,
    *,
    checkpoint_id: str,
    decision: str,
    instructions: str | None = None,
    modified_payload: str | None = None,
) -> CheckpointRecord:
    """Resolve a pending checkpoint with an operator decision.

    Q&A is *not* a resolution — use :func:`add_question` for that.
    Returns the updated record.
    """
    if decision not in _VALID_DECISIONS:
        raise ValueError(f"invalid decision: {decision}")
    op_input: dict[str, Any] = {"decision": decision}
    if instructions is not None and instructions != "":
        op_input["instructions"] = instructions
    if modified_payload is not None and modified_payload != "":
        op_input["modified_payload"] = modified_payload

    cur = conn.execute(
        "UPDATE checkpoints SET status = ?, operator_input = ?, resolved_at = ? "
        "WHERE checkpoint_id = ? AND status = 'pending'",
        (_STATUS_FOR_DECISION[decision], json.dumps(op_input), now_iso(), checkpoint_id),
    )
    if cur.rowcount != 1:
        existing = get_checkpoint(conn, checkpoint_id)
        if existing is None:
            raise ValueError(f"checkpoint not found: {checkpoint_id}")
        raise RuntimeError(
            f"checkpoint {checkpoint_id} is not pending (status={existing.status})"
        )
    rec = get_checkpoint(conn, checkpoint_id)
    assert rec is not None
    return rec


def resolve_payload_for_resume(cp: CheckpointRecord) -> str:
    """Compute the next-agent payload from the operator's decision.

    approve  → default payload unchanged
    modify   → ``modified_payload`` if given, else default + appended instructions
    reject   → caller must not resume; raises
    """
    if cp.status == "rejected":
        raise RuntimeError(f"checkpoint {cp.checkpoint_id} was rejected — cannot resume")
    if cp.status not in {"approved", "modified"}:
        raise RuntimeError(
            f"checkpoint {cp.checkpoint_id} is not resolved (status={cp.status})"
        )
    default = cp.default_payload
    if cp.status == "approved":
        return default
    op = cp.operator_input or {}
    override = op.get("modified_payload")
    if override:
        return override
    instructions = op.get("instructions") or ""
    if instructions:
        return f"{default}\n\nOPERATOR_INSTRUCTIONS: {instructions}"
    return default


# ---------------------------------------------------------------------------
# Q&A — operator can converse without resolving the checkpoint.
#
# Phase 4 persists the thread; an LLM-backed answer pass (run a one-shot
# Claude call against the run history + checkpoint payload + question)
# is wired into the Streamlit UI but kept manual: the operator chooses
# when to spend the dollars, and the smoke test exercises the persistence
# path without touching the network.
# ---------------------------------------------------------------------------

def add_question(
    conn: sqlite3.Connection,
    *,
    checkpoint_id: str,
    role: str,
    content: str,
    agent_used: str | None = None,
    cost_usd: float | None = None,
) -> str:
    if role not in {"operator", "assistant"}:
        raise ValueError(f"invalid role: {role}")
    qid = new_id("q_")
    conn.execute(
        "INSERT INTO checkpoint_questions "
        "(question_id, checkpoint_id, role, content, created_at, agent_used, cost_usd) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (qid, checkpoint_id, role, content, now_iso(), agent_used, cost_usd),
    )
    return qid


def list_questions(conn: sqlite3.Connection, checkpoint_id: str) -> list[QuestionRecord]:
    rows = conn.execute(
        "SELECT * FROM checkpoint_questions WHERE checkpoint_id = ? "
        "ORDER BY created_at ASC",
        (checkpoint_id,),
    ).fetchall()
    return [
        QuestionRecord(
            question_id=r["question_id"],
            checkpoint_id=r["checkpoint_id"],
            role=r["role"],
            content=r["content"],
            created_at=r["created_at"],
            agent_used=r["agent_used"],
            cost_usd=r["cost_usd"],
        )
        for r in rows
    ]
