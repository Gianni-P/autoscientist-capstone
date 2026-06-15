"""Tests for the code_review CP3 gate and revision-loop cap.

Coverage:
  * ``stage_for_agent`` returns stage 3 only on a forward handoff.
  * ``_max_code_review_cycles`` env override path.
  * Runner opens CP3 when code_review passes (forward branch).
  * Runner forces CP3 with ``extra.loop_cap_exceeded`` after N revisions.
  * Runner does not open CP3 on the first revise of a normal loop.
  * Resume after a forced CP3 resets the cycle counter and re-enters the loop.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.clients import mock as mock_client


# ---------------------------------------------------------------------------
# Pure-unit: stage_for_agent
# ---------------------------------------------------------------------------

def test_stage_for_agent_revise_loop_returns_none() -> None:
    """code_review handing back to code_gen is the revise loop — no CP."""
    assert cp_manager.stage_for_agent("code_review", handoff_to="code_gen") is None


def test_stage_for_agent_forward_returns_stage3() -> None:
    info = cp_manager.stage_for_agent("code_review", handoff_to="results_validator")
    assert info == (3, "preliminary_review")


def test_stage_for_agent_other_agents_ignore_handoff_to() -> None:
    """The handoff_to gate only applies to code_review."""
    assert cp_manager.stage_for_agent("methodology", handoff_to="anything") == (
        2, "methodology_approval",
    )
    assert cp_manager.stage_for_agent("peer_reviewer", handoff_to="paper_writer") == (
        5, "draft_review",
    )


def test_stage_for_agent_unknown_returns_none() -> None:
    assert cp_manager.stage_for_agent("idea_gen") is None
    assert cp_manager.stage_for_agent("nope", handoff_to="anything") is None


# ---------------------------------------------------------------------------
# Pure-unit: _max_code_review_cycles
# ---------------------------------------------------------------------------

def test_max_code_review_cycles_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoscientist.runtime.config import load_config
    from autoscientist.runtime.runner import _max_code_review_cycles

    monkeypatch.setenv("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES", "7")
    cfg = load_config()
    assert _max_code_review_cycles(cfg) == 7


def test_max_code_review_cycles_invalid_env_falls_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoscientist.runtime.config import load_config
    from autoscientist.runtime.runner import (
        DEFAULT_MAX_CODE_REVIEW_CYCLES,
        _max_code_review_cycles,
    )

    cfg = load_config()
    monkeypatch.setenv("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES", "not-an-int")
    assert _max_code_review_cycles(cfg) == DEFAULT_MAX_CODE_REVIEW_CYCLES
    monkeypatch.setenv("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES", "0")
    assert _max_code_review_cycles(cfg) == DEFAULT_MAX_CODE_REVIEW_CYCLES


# ---------------------------------------------------------------------------
# Integration: runner + mock provider
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Per-test isolated DB + runs dir + projects dir + mock-routed agents."""
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

    for name in ("code_gen", "test_gen", "code_review",
                 "results_validator", "paper_writer", "peer_reviewer"):
        cfg.models["agents"][name]["model"] = "mock_stub"

    yield cfg

    reset_for_tests()


def _open_db(cfg) -> sqlite3.Connection:
    from autoscientist.state.db import open_db
    return open_db(cfg.db_path())


def test_runner_opens_cp3_when_code_review_passes(runner_env) -> None:
    """The canonical mock code_review fixture returns verdict=pass, handing
    forward to results_validator. The runner should open a CP3 there."""
    from autoscientist.runtime.runner import run

    cfg = runner_env
    run_id = run(
        starting_agent="code_review",
        project_id="t1",
        initial_payload=json.dumps({"src_files": [], "test_files": []}),
        enable_checkpoints=True,
        max_handoffs=20,
        cfg=cfg,
    )

    conn = _open_db(cfg)
    try:
        row = conn.execute(
            "SELECT status, note FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert row["status"] == "paused"
        cps = cp_manager.list_for_run(conn, run_id)
        assert len(cps) == 1
        cp = cps[0]
        assert cp.stage == 3
        assert cp.from_agent == "code_review"
        assert cp.to_agent == "results_validator"
        # No loop-cap metadata on a clean forward pass.
        assert cp.extra is None
    finally:
        conn.close()


def test_runner_does_not_open_cp3_on_first_revise(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First revise from code_review must not open CP3 — only the loop-cap
    forces it. Sets cap=10 so a single iteration finishes naturally and the
    chain returns to code_gen without a pause."""
    from autoscientist.runtime.runner import run

    monkeypatch.setenv("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES", "10")

    def _revise(inbound: str) -> str:
        return (
            json.dumps({"findings": [], "verdict": "revise", "summary": "mock revise"})
            + "\n\nHANDOFF: code_gen\n"
            + json.dumps({"findings": [], "instruction": "redo"})
        )

    monkeypatch.setattr(mock_client, "_fix_code_review", _revise)
    mock_client._FIXTURES["code_review"] = _revise

    cfg = runner_env
    # Cap the handoff count tight so the run stops quickly without piling up.
    run_id = run(
        starting_agent="code_review",
        project_id="t2",
        initial_payload=json.dumps({"src_files": [], "test_files": []}),
        enable_checkpoints=True,
        max_handoffs=4,
        cfg=cfg,
    )

    conn = _open_db(cfg)
    try:
        cps = cp_manager.list_for_run(conn, run_id)
        # No CP3 should have opened during the first few revise iterations.
        stage3_cps = [c for c in cps if c.stage == 3]
        assert stage3_cps == []
    finally:
        conn.close()


def test_runner_loop_cap_forces_cp3_after_n_revisions(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With max_code_review_cycles=2 and a mock that always revises, the
    runner must force a CP3 carrying extra.loop_cap_exceeded=True after
    the 2nd code_review iteration."""
    from autoscientist.runtime.runner import run

    monkeypatch.setenv("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES", "2")

    def _revise(inbound: str) -> str:
        return (
            json.dumps({"findings": [{"severity": "major", "issue": "mock"}],
                        "verdict": "revise", "summary": "mock revise loop"})
            + "\n\nHANDOFF: code_gen\n"
            + json.dumps({"findings": [], "instruction": "redo"})
        )

    monkeypatch.setattr(mock_client, "_fix_code_review", _revise)
    mock_client._FIXTURES["code_review"] = _revise

    cfg = runner_env
    run_id = run(
        starting_agent="code_review",
        project_id="t3",
        initial_payload=json.dumps({"src_files": [], "test_files": []}),
        enable_checkpoints=True,
        max_handoffs=50,
        cfg=cfg,
    )

    conn = _open_db(cfg)
    try:
        row = conn.execute(
            "SELECT status, note FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert row["status"] == "paused"
        cps = cp_manager.list_for_run(conn, run_id)
        stage3_cps = [c for c in cps if c.stage == 3]
        assert len(stage3_cps) == 1
        cp = stage3_cps[0]
        assert cp.from_agent == "code_review"
        assert cp.to_agent == "code_gen"  # the loop wants another revise
        assert cp.extra is not None
        assert cp.extra["loop_cap_exceeded"] is True
        assert cp.extra["cycles"] == 2
        assert cp.extra["max_cycles"] == 2
    finally:
        conn.close()


def test_runner_resume_after_loop_cap_resets_counter(
    runner_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the operator approves the loop-cap CP3, _drive_loop starts
    fresh with code_review_cycles=0.

    Verified indirectly by re-running the same revise-only chain: a fresh
    counter must permit two more code_review firings before re-firing the
    cap (rather than the cap firing immediately or accumulating).

    The runner cache short-circuits attempts to make the mock stateful
    (test_gen + code_review inputs are deterministic across iterations),
    so this test asserts on the cycles field of the *new* CP3 instead of
    on a verdict transition.
    """
    from autoscientist.runtime.runner import resume_run, run

    monkeypatch.setenv("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES", "2")

    def _revise(inbound: str) -> str:
        return (
            json.dumps({"findings": [], "verdict": "revise", "summary": "mock revise"})
            + "\n\nHANDOFF: code_gen\n"
            + json.dumps({"instruction": "redo"})
        )

    monkeypatch.setattr(mock_client, "_fix_code_review", _revise)
    mock_client._FIXTURES["code_review"] = _revise

    cfg = runner_env
    run_id = run(
        starting_agent="code_review",
        project_id="t4",
        initial_payload=json.dumps({"src_files": [], "test_files": []}),
        enable_checkpoints=True,
        max_handoffs=50,
        cfg=cfg,
    )

    conn = _open_db(cfg)
    try:
        cps_after_first = cp_manager.list_for_run(conn, run_id)
        assert len(cps_after_first) == 1
        cp_loop = cps_after_first[0]
        assert cp_loop.stage == 3
        assert cp_loop.extra is not None
        assert cp_loop.extra["loop_cap_exceeded"] is True
        assert cp_loop.extra["cycles"] == 2

        cp_manager.resolve(
            conn,
            checkpoint_id=cp_loop.checkpoint_id,
            decision=cp_manager.DECISION_APPROVE,
        )
        conn.commit()
    finally:
        conn.close()

    resume_run(run_id, cfg=cfg)

    conn = _open_db(cfg)
    try:
        cps_after_resume = cp_manager.list_for_run(conn, run_id)
        stage3_cps = [c for c in cps_after_resume if c.stage == 3]
        # Counter reset means we get a fresh cap-fire after another 2 cycles,
        # not at cycles=4. Two stage-3 records, each at cycles=2.
        assert len(stage3_cps) == 2
        for cp in stage3_cps:
            assert cp.extra is not None
            assert cp.extra["loop_cap_exceeded"] is True
            assert cp.extra["cycles"] == 2, (
                f"counter did not reset on resume — got cycles={cp.extra['cycles']}"
            )
    finally:
        conn.close()
