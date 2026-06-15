"""Runner-level integration test for the files: [...] safety net.

Drives the runner through ``code_gen`` against the mock provider and
asserts the safety net wrote each mock fixture file into the sandbox,
and that the JSONL log contains a ``run.payload_files_persisted`` event.

This is the only test that exercises the path through ``runtime.runner``
itself rather than calling ``persist_files_from_payload`` directly.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create an isolated DB/runs/projects setup and return the projects root.

    The runner derives paths from the global config; we point
    ``AUTOSCIENTIST_DB_PATH`` at a temp DB and override the cached config's
    ``paths.runs_dir`` and ``paths.projects_dir`` to live under tmp_path.
    """
    from autoscientist.runtime.config import load_config, reset_for_tests

    monkeypatch.setenv("AUTOSCIENTIST_DB_PATH", str(tmp_path / "test.db"))
    reset_for_tests()
    cfg = load_config(reload=True)

    runs_dir = tmp_path / "runs"
    projects_dir = tmp_path / "projects"
    runs_dir.mkdir()
    projects_dir.mkdir()
    # Absolute paths: ``cfg.runs_dir()`` does ``self.root / rel``; an absolute
    # ``rel`` makes pathlib discard ``self.root``, so the override sticks.
    cfg.default.setdefault("paths", {})["runs_dir"] = str(runs_dir)
    cfg.default["paths"]["projects_dir"] = str(projects_dir)

    # Force code_gen + downstream agents to the mock provider.
    for name in ("code_gen", "test_gen", "code_review",
                 "results_validator", "paper_writer", "peer_reviewer"):
        cfg.models["agents"][name]["model"] = "mock_stub"

    yield cfg

    # Cleanup global config so other tests start fresh.
    reset_for_tests()


def test_runner_persists_files_from_code_gen_mock_payload(
    isolated_run_env, tmp_path: Path,
) -> None:
    """End-to-end: starting at code_gen with the mock provider, the runner
    should write each file from the fixture's ``files: [...]`` array into
    the project sandbox."""
    from autoscientist.runtime.runner import run

    cfg = isolated_run_env
    project_id = "smoke_payload_files"

    # Run a short chain; let it terminate or hit max_handoffs naturally.
    run_id = run(
        starting_agent="code_gen",
        project_id=project_id,
        initial_payload=json.dumps({"plan": {}, "first_step": "scaffold"}),
        enable_checkpoints=False,
        max_handoffs=8,
        cfg=cfg,
    )

    # Sandbox should contain every file the mock code_gen fixture promised.
    projects_root = cfg.root / cfg.default["paths"]["projects_dir"]
    sandbox = projects_root / project_id / "sandbox"
    assert (sandbox / "src" / "data.py").read_text(encoding="utf-8") == "# mock data loader\n"
    assert (sandbox / "src" / "train.py").read_text(encoding="utf-8") == "# mock train loop\n"
    assert (sandbox / "scripts" / "run.sh").read_text(encoding="utf-8").startswith(
        "#!/usr/bin/env bash"
    )

    # JSONL log must record the safety-net firing for code_gen.
    log_path = cfg.runs_dir() / run_id / "logs" / "run.jsonl"
    assert log_path.exists()
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    persisted_events = [
        e for e in events
        if e.get("event") == "run.payload_files_persisted" and e.get("agent") == "code_gen"
    ]
    assert len(persisted_events) >= 1, "expected at least one persisted event for code_gen"
    first = persisted_events[0]
    assert first["n_ok"] == 3
    assert first["n_error"] == 0
    paths = set(first["paths"])
    assert {"src/data.py", "src/train.py", "scripts/run.sh"}.issubset(paths)
