"""Regression tests for the M27 / M3 / M7 audit follow-ups."""

from __future__ import annotations

import os
from contextlib import closing

from autoscientist.runtime.runner import _norm_agent_name, _resolve_off_topology
from autoscientist.state.db import open_db
from autoscientist.tools import tool_cache

# --- M27: tool_cache.cache_get is a non-committing read ----------------------

def test_tool_cache_get_does_not_commit_callers_transaction(tmp_path):
    with closing(open_db(tmp_path / "tc.db")) as conn:
        tool_cache.cache_put(conn, "t", "k", {"v": 1})  # commits the stored entry
        assert not conn.in_transaction
        # Open an uncommitted transaction on the same connection (a no-op DML).
        conn.execute("UPDATE tool_cache SET hit_count = hit_count WHERE 1 = 0")
        assert conn.in_transaction
        # A cache hit must return the value WITHOUT committing the caller's
        # in-flight transaction (the old code called conn.commit() here).
        assert tool_cache.cache_get(conn, "t", "k") == {"v": 1}
        assert conn.in_transaction


def test_tool_cache_get_miss_returns_none(tmp_path):
    with closing(open_db(tmp_path / "tc.db")) as conn:
        assert tool_cache.cache_get(conn, "t", "absent") is None


# --- M3: off-topology handoff correction -------------------------------------

_CODE_REVIEW_TARGETS = ("code_gen", "results_validator")


def test_norm_agent_name_collapses_punctuation_and_case():
    assert _norm_agent_name("Code-Gen") == "codegen"
    assert _norm_agent_name("code_gen") == "codegen"
    assert _norm_agent_name("code gen") == "codegen"


def test_off_topology_snaps_formatting_variant_to_allowed_target():
    # A typo/format variant of an allowed target snaps back to it.
    assert _resolve_off_topology("code_review", "Code-Gen", _CODE_REVIEW_TARGETS) == "code_gen"
    assert _resolve_off_topology("code_review", "results-validator", _CODE_REVIEW_TARGETS) == "results_validator"


def test_off_topology_genuinely_bogus_target_redirects_to_forward_stage():
    # 'paper_writer' is not an allowed code_review target -> forward stage.
    assert _resolve_off_topology("code_review", "paper_writer", _CODE_REVIEW_TARGETS) == "results_validator"


def test_off_topology_terminal_agent_returns_none():
    # peer_reviewer has no forward stage -> end the run rather than chase it.
    assert _resolve_off_topology("peer_reviewer", "anything", ("repo_publisher", "paper_writer")) is None


def test_off_topology_never_returns_the_bogus_target():
    out = _resolve_off_topology("code_gen", "some_hallucination", ("test_gen",))
    assert out != "some_hallucination"
    assert out == "test_gen"  # code_gen's forward stage


# --- M7: resume runs as a detached subprocess --------------------------------

def test_resume_command_targets_runner_resume_cli():
    from autoscientist.checkpoints import ui

    cmd = ui._resume_command("run_123")
    assert cmd[0]  # the interpreter (sys.executable)
    assert cmd[1:] == ["-m", "autoscientist.runtime.runner", "--resume", "run_123"]


def test_resume_in_background_spawns_detached_subprocess(monkeypatch):
    from autoscientist.checkpoints import ui

    captured: dict = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr(ui.subprocess, "Popen", _FakePopen)
    ui._resume_in_background("run_xyz")

    assert captured["cmd"][-2:] == ["--resume", "run_xyz"]
    # Detached from the UI: dedicated stdio + (posix) a new session so the child
    # survives the Streamlit process exiting.
    assert captured["kwargs"].get("stdin") is not None
    assert captured["kwargs"].get("stdout") is not None
    if os.name == "posix":
        assert captured["kwargs"].get("start_new_session") is True


def test_ui_module_is_importable_without_running_main():
    # Guarding the main() call means importing the module for tests doesn't
    # render the whole Streamlit page (which would need a script-run context).
    from autoscientist.checkpoints import ui

    assert hasattr(ui, "main")
    assert ui._running_under_streamlit() is False  # not under `streamlit run` here
