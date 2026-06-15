"""Tests for manual pause / resume.

Covers ``runtime.control`` helpers in isolation and the end-to-end path
through the runner (request pause → runner honours at next boundary →
saved state on disk → resume rebuilds counters → run continues).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.clients import mock as mock_client
from autoscientist.runtime import control as run_control


# ---------------------------------------------------------------------------
# runtime.control helpers (pure unit)
# ---------------------------------------------------------------------------

def _fresh_db(tmp_path: Path) -> sqlite3.Connection:
    from autoscientist.state.db import open_db
    conn = open_db(tmp_path / "ctl.db")
    conn.execute(
        "INSERT INTO runs (run_id, project_id, status, started_at) "
        "VALUES ('run_x', 'proj', 'running', '2026-05-18T00:00:00.000+00:00')"
    )
    conn.commit()
    return conn


def test_request_pause_creates_row(tmp_path: Path) -> None:
    with _fresh_db(tmp_path) as conn:
        run_control.request_pause(conn, "run_x")
        state = run_control.read_pause_state(conn, "run_x")
        assert state is not None
        assert state.pause_requested is True
        assert state.requested_at is not None
        assert state.paused_at is None
        assert state.next_agent is None


def test_request_pause_is_idempotent(tmp_path: Path) -> None:
    with _fresh_db(tmp_path) as conn:
        run_control.request_pause(conn, "run_x")
        first = run_control.read_pause_state(conn, "run_x")
        # Second click before the runner honours it — must not overwrite the
        # original timestamp.
        run_control.request_pause(conn, "run_x")
        second = run_control.read_pause_state(conn, "run_x")
        assert second.requested_at == first.requested_at
        assert second.pause_requested is True


def test_is_pause_requested_returns_false_when_no_row(tmp_path: Path) -> None:
    with _fresh_db(tmp_path) as conn:
        assert run_control.is_pause_requested(conn, "run_x") is False


def test_save_pause_state_then_read(tmp_path: Path) -> None:
    with _fresh_db(tmp_path) as conn:
        run_control.request_pause(conn, "run_x")
        run_control.save_pause_state(
            conn,
            run_id="run_x",
            next_agent="code_gen",
            next_payload='{"instr": "redo"}',
            handoffs_so_far=4,
            code_review_cycles=2,
        )
        state = run_control.read_pause_state(conn, "run_x")
        assert state.is_active
        assert state.pause_requested is False
        assert state.next_agent == "code_gen"
        assert state.next_payload == '{"instr": "redo"}'
        assert state.handoffs_so_far == 4
        assert state.code_review_cycles == 2
        assert state.paused_at is not None


def test_clear_pause_state(tmp_path: Path) -> None:
    with _fresh_db(tmp_path) as conn:
        run_control.save_pause_state(
            conn,
            run_id="run_x",
            next_agent="code_gen",
            next_payload="",
            handoffs_so_far=0,
            code_review_cycles=0,
        )
        assert run_control.read_pause_state(conn, "run_x") is not None
        run_control.clear_pause_state(conn, "run_x")
        assert run_control.read_pause_state(conn, "run_x") is None


def test_cancel_pause_request_only_clears_unhonoured(tmp_path: Path) -> None:
    with _fresh_db(tmp_path) as conn:
        run_control.request_pause(conn, "run_x")
        run_control.cancel_pause_request(conn, "run_x")
        state = run_control.read_pause_state(conn, "run_x")
        assert state is not None
        assert state.pause_requested is False

        # Now save state (simulating the runner honouring a pause) and
        # try to cancel — must NOT wipe the saved state.
        run_control.save_pause_state(
            conn, run_id="run_x", next_agent="code_gen",
            next_payload="", handoffs_so_far=0, code_review_cycles=0,
        )
        run_control.cancel_pause_request(conn, "run_x")
        state = run_control.read_pause_state(conn, "run_x")
        assert state is not None
        assert state.is_active
        assert state.next_agent == "code_gen"


# ---------------------------------------------------------------------------
# Runner integration: pause honoured + manual resume
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from autoscientist.runtime.config import load_config, reset_for_tests

    monkeypatch.setenv("AUTOSCIENTIST_DB_PATH", str(tmp_path / "test.db"))
    reset_for_tests()
    cfg = load_config(reload=True)

    runs_dir = tmp_path / "runs"
    projects_dir = tmp_path / "projects"
    runs_dir.mkdir()
    projects_dir.mkdir()
    cfg.default.setdefault("paths", {})["runs_dir"] = str(runs_dir)
    cfg.default["paths"]["projects_dir"] = str(projects_dir)

    for name in ("idea_gen", "idea_critic", "methodology", "code_gen",
                 "test_gen", "code_review", "results_validator",
                 "paper_writer", "peer_reviewer"):
        cfg.models["agents"][name]["model"] = "mock_stub"

    yield cfg

    reset_for_tests()


def _open(cfg) -> sqlite3.Connection:
    from autoscientist.state.db import open_db
    return open_db(cfg.db_path())


def test_runner_honours_manual_pause_between_agents(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set pause_requested before the run starts; the runner should
    honour it at the first agent boundary (after idea_gen) and stop
    before reaching idea_critic."""
    from autoscientist.runtime.runner import run

    cfg = runner_env

    # Pre-create the runs row trick won't work — runs are created inside
    # run(). Instead, monkeypatch _invoke_agent to flip the pause flag
    # immediately after idea_gen finishes its first turn.
    from autoscientist.runtime import runner as runner_mod
    original_invoke = runner_mod._invoke_agent

    def _flip_after_idea_gen(*, conn, agent, **kw):
        result = original_invoke(conn=conn, agent=agent, **kw)
        if agent.name == "idea_gen":
            run_control.request_pause(conn, kw["run_id"])
            conn.commit()
        return result

    monkeypatch.setattr(runner_mod, "_invoke_agent", _flip_after_idea_gen)

    run_id = run(
        starting_agent="idea_gen",
        project_id="pause_t1",
        initial_payload=json.dumps({"direction": "test"}),
        enable_checkpoints=False,
        max_handoffs=20,
        cfg=cfg,
    )

    with _open(cfg) as conn:
        status = conn.execute(
            "SELECT status, note FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert status["status"] == "paused"
        assert status["note"] == "manual_pause"

        pause = run_control.read_pause_state(conn, run_id)
        assert pause is not None
        assert pause.is_active
        # idea_gen → idea_critic per the mock chain.
        assert pause.next_agent == "idea_critic"
        assert pause.handoffs_so_far == 0  # incremented AFTER this point
        # Sanity check: idea_critic should NOT have run.
        critic_msgs = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'idea_critic'",
            (run_id,),
        ).fetchone()["n"]
        assert critic_msgs == 0


def test_runner_resume_from_manual_pause_continues_chain(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: pause after idea_gen → resume_run → idea_critic
    actually fires and lands a message."""
    from autoscientist.runtime.runner import resume_run, run
    from autoscientist.runtime import runner as runner_mod

    cfg = runner_env

    original_invoke = runner_mod._invoke_agent

    def _flip_after_idea_gen(*, conn, agent, **kw):
        result = original_invoke(conn=conn, agent=agent, **kw)
        if agent.name == "idea_gen":
            run_control.request_pause(conn, kw["run_id"])
            conn.commit()
        return result

    monkeypatch.setattr(runner_mod, "_invoke_agent", _flip_after_idea_gen)

    run_id = run(
        starting_agent="idea_gen",
        project_id="pause_t2",
        initial_payload=json.dumps({"direction": "test"}),
        enable_checkpoints=False,
        max_handoffs=20,
        cfg=cfg,
    )

    # Restore original behaviour for the resume — no more auto-pause.
    monkeypatch.setattr(runner_mod, "_invoke_agent", original_invoke)

    resume_run(run_id, cfg=cfg)

    with _open(cfg) as conn:
        # The pause row was cleared.
        assert run_control.read_pause_state(conn, run_id) is None
        # idea_critic ran post-resume.
        critic_msgs = conn.execute(
            "SELECT COUNT(*) AS n FROM messages "
            "WHERE run_id = ? AND agent_name = 'idea_critic'",
            (run_id,),
        ).fetchone()["n"]
        assert critic_msgs > 0
        # The manual_resume handoff record exists.
        resume_msgs = conn.execute(
            "SELECT content FROM messages "
            "WHERE run_id = ? AND role = 'handoff' "
            "ORDER BY created_at ASC",
            (run_id,),
        ).fetchall()
        assert any('"kind": "manual_resume"' in r["content"] for r in resume_msgs)


def test_pause_requested_but_checkpoint_fires_first_does_not_double_pause(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a CP fires while a pause is pending, the CP wins. The pause
    flag stays set; ``cancel_pause_request`` on resume from the CP
    clears it before the chain continues.

    Set the flag right after ``idea_critic`` (which is in CHECKPOINT_POLICY).
    The runner then checks the checkpoint condition first and opens CP1,
    leaving ``pause_requested=1`` un-honoured in the DB."""
    from autoscientist.runtime.runner import run
    from autoscientist.runtime import runner as runner_mod

    cfg = runner_env

    original_invoke = runner_mod._invoke_agent

    def _flip_after_idea_critic(*, conn, agent, **kw):
        result = original_invoke(conn=conn, agent=agent, **kw)
        if agent.name == "idea_critic":
            run_control.request_pause(conn, kw["run_id"])
            conn.commit()
        return result

    monkeypatch.setattr(runner_mod, "_invoke_agent", _flip_after_idea_critic)

    run_id = run(
        starting_agent="idea_gen",
        project_id="pause_t3",
        initial_payload=json.dumps({"direction": "test"}),
        enable_checkpoints=True,  # let CP1 actually fire
        max_handoffs=20,
        cfg=cfg,
    )

    with _open(cfg) as conn:
        status = conn.execute(
            "SELECT status, note FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert status["status"] == "paused"
        pending = [
            c for c in cp_manager.list_for_run(conn, run_id)
            if c.status == "pending"
        ]
        # CP1 (idea_selection) opened from idea_critic — checkpoint wins
        # over the pause poll which sits below it in the loop.
        assert any(c.stage == 1 for c in pending), (
            f"expected CP1 to win, got pending="
            f"{[(c.stage, c.from_agent) for c in pending]}"
        )
        # The pause request is recorded but never honoured (no paused_at).
        pause = run_control.read_pause_state(conn, run_id)
        assert pause is not None
        assert pause.pause_requested is True
        assert pause.paused_at is None


def test_resume_after_cp_clears_stale_pause_request(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the operator approves a CP-resume, the stale pause request
    should be cleared so the next ``_drive_loop`` doesn't immediately
    honour a pause the operator forgot about."""
    from autoscientist.runtime.runner import resume_run, run
    from autoscientist.runtime import runner as runner_mod

    cfg = runner_env
    original_invoke = runner_mod._invoke_agent

    def _flip_after_idea_critic(*, conn, agent, **kw):
        result = original_invoke(conn=conn, agent=agent, **kw)
        if agent.name == "idea_critic":
            run_control.request_pause(conn, kw["run_id"])
            conn.commit()
        return result

    monkeypatch.setattr(runner_mod, "_invoke_agent", _flip_after_idea_critic)
    run_id = run(
        starting_agent="idea_gen",
        project_id="pause_t4",
        initial_payload=json.dumps({"direction": "test"}),
        enable_checkpoints=True,
        max_handoffs=20,
        cfg=cfg,
    )
    monkeypatch.setattr(runner_mod, "_invoke_agent", original_invoke)

    # Approve the pending CP1 the operator would do via the UI.
    with _open(cfg) as conn:
        pending = [c for c in cp_manager.list_for_run(conn, run_id)
                   if c.status == "pending"]
        assert pending
        cp_manager.resolve(
            conn, checkpoint_id=pending[0].checkpoint_id,
            decision=cp_manager.DECISION_APPROVE,
        )
        conn.commit()

    resume_run(run_id, cfg=cfg)

    with _open(cfg) as conn:
        pause = run_control.read_pause_state(conn, run_id)
        # Either the row was removed (manual-pause never honoured + cleared
        # by cancel_pause_request) or pause_requested is now 0.
        assert pause is None or not pause.pause_requested
