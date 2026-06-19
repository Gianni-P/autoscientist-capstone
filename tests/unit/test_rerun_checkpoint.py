"""Operator 're-run with nudge' at a checkpoint.

Covers the ``DECISION_RERUN`` decision (cp_manager), the nudge helpers, and
the ``resume_run`` rerun branch that re-invokes the agent that produced the
checkpoint on its original inbound + nudge (instead of advancing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.runtime import runner
from autoscientist.runtime.config import Config
from autoscientist.state.db import open_db, record_message, start_run

# ---------------------------------------------------------------------------
# cp_manager: the new decision
# ---------------------------------------------------------------------------

def test_rerun_is_a_valid_decision_mapping_to_modified_status(tmp_path: Path) -> None:
    assert cp_manager.DECISION_RERUN in cp_manager._VALID_DECISIONS
    assert cp_manager._STATUS_FOR_DECISION[cp_manager.DECISION_RERUN] == "modified"
    with open_db(tmp_path / "t.db") as conn:
        run_id = start_run(conn, "p")
        cp_id = cp_manager.open_checkpoint(
            conn, run_id=run_id, stage=1, from_agent="idea_critic",
            to_agent="methodology", agent_output_raw="{}", default_payload="DEF",
        )
        rec = cp_manager.resolve(
            conn, checkpoint_id=cp_id,
            decision=cp_manager.DECISION_RERUN, instructions="focus on X",
        )
        assert rec.status == "modified"
        assert rec.operator_input["decision"] == "rerun"
        assert rec.operator_input["instructions"] == "focus on X"


# ---------------------------------------------------------------------------
# nudge helpers
# ---------------------------------------------------------------------------

def test_apply_nudge_appends_and_replaces() -> None:
    once = runner._apply_nudge("BASE INPUT", "do better")
    assert once == "BASE INPUT\n\nOPERATOR_NUDGE: do better"
    # a second nudge replaces (does not stack) the first
    twice = runner._apply_nudge(once, "now do this")
    assert twice == "BASE INPUT\n\nOPERATOR_NUDGE: now do this"
    assert twice.count("OPERATOR_NUDGE") == 1


def test_apply_nudge_empty_returns_base() -> None:
    assert runner._apply_nudge("BASE", "") == "BASE"
    assert runner._apply_nudge(None, "x").endswith("OPERATOR_NUDGE: x")


# ---------------------------------------------------------------------------
# resume_run rerun branch (drive loop stubbed — no API calls)
# ---------------------------------------------------------------------------

def _cfg_for(tmp_path: Path) -> Config:
    return Config(
        default={"paths": {"db_path": "rerun.db", "runs_dir": "runs"},
                 "runtime": {"default_max_handoffs": 50}},
        models={}, mcp={}, root=tmp_path,
    )


def test_resume_run_rerun_reinvokes_from_agent_with_nudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg_for(tmp_path)
    conn = open_db(cfg.db_path())
    run_id = start_run(conn, "proj")
    # The agent's original inbound (what re-run must replay).
    record_message(conn, run_id=run_id, agent_name="idea_critic", role="user",
                   content="ORIGINAL INBOUND")
    record_message(conn, run_id=run_id, agent_name="idea_critic", role="assistant",
                   content="critic output\nHANDOFF: methodology")
    cp_id = cp_manager.open_checkpoint(
        conn, run_id=run_id, stage=1, from_agent="idea_critic",
        to_agent="methodology", agent_output_raw="{}", default_payload="DEF",
    )
    cp_manager.resolve(conn, checkpoint_id=cp_id,
                       decision=cp_manager.DECISION_RERUN, instructions="be rigorous")
    conn.execute("UPDATE runs SET status='paused' WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()

    captured: dict = {}

    def fake_drive_loop(**kwargs):
        captured.update(kwargs)
        return ("paused", "awaiting operator", 0)

    monkeypatch.setattr(runner, "_drive_loop", fake_drive_loop)

    out = runner.resume_run(run_id, cfg=cfg)
    assert out == run_id
    # Re-ran the producing agent, not the next one:
    assert captured["starting_agent"] == "idea_critic"
    assert "ORIGINAL INBOUND" in captured["starting_payload"]
    assert "OPERATOR_NUDGE: be rigorous" in captured["starting_payload"]

    # A handoff audit row records the rerun decision.
    with open_db(cfg.db_path()) as conn2:
        rows = conn2.execute(
            "SELECT content FROM messages WHERE run_id=? AND role='handoff'", (run_id,)
        ).fetchall()
    assert any('"decision": "rerun"' in r["content"] for r in rows)
