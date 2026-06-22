"""figure_gen agent: registry resolution, pipeline topology, and a mock smoke.

figure_gen is the new agent inserted between results_validator and paper_writer.
It reads the validated results, writes + runs a matplotlib plot script, and hands
the figure paths/captions to paper_writer. These tests pin:
  * the agent resolves with the right tools + handoff targets + a real prompt;
  * the topology is wired both ways (results_validator -> figure_gen ->
    paper_writer) across handoff_targets, _FORWARD_TARGET, and PIPELINE_ORDER;
  * it opens NO new operator checkpoint (rides inside the CP4->CP5 leg);
  * it is orchestratable (both the runtime set and the manager mirror);
  * the mock chain actually routes results_validator -> figure_gen -> paper_writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoscientist.agents import AGENT_NAMES, get_agent
from autoscientist.checkpoints import manager as cp_manager
from autoscientist.runtime import orchestration, runner
from autoscientist.runtime.agent import load_prompt
from autoscientist.runtime.config import load_config


# ---------------------------------------------------------------------------
# Registry + prompt
# ---------------------------------------------------------------------------

def test_figure_gen_listed_in_agent_names() -> None:
    assert "figure_gen" in AGENT_NAMES


def test_figure_gen_resolves_with_real_prompt() -> None:
    cfg = load_config()
    agent = get_agent("figure_gen", cfg)
    assert agent is not None
    assert agent.name == "figure_gen"
    # Forward to paper_writer first (forward-first ordering), revise back to
    # results_validator.
    assert agent.handoff_targets[0] == "paper_writer"
    assert "results_validator" in agent.handoff_targets
    # Hybrid toolset: code-writing tools PLUS execute (to render) + sandbox reads.
    for tool in ("write_file", "execute", "read_sandbox_file",
                 "list_sandbox", "check_imports", "handoff"):
        assert tool in agent.tools, f"figure_gen missing tool {tool}"
    # Prompt exists and loads with frontmatter.
    assert agent.system_prompt_path.exists()
    prompt = load_prompt(agent.system_prompt_path)
    assert "figure" in prompt.system_text.lower()


def test_figure_gen_routing_in_models_toml() -> None:
    cfg = load_config()
    agents = cfg.models.get("agents", {})
    assert "figure_gen" in agents
    assert agents["figure_gen"]["model"]


# ---------------------------------------------------------------------------
# Topology: results_validator -> figure_gen -> paper_writer
# ---------------------------------------------------------------------------

def test_results_validator_hands_off_to_figure_gen() -> None:
    cfg = load_config()
    rv = get_agent("results_validator", cfg)
    assert rv is not None
    assert rv.handoff_targets[0] == "figure_gen"  # forward target
    assert "code_gen" in rv.handoff_targets       # revise target preserved


def test_forward_target_threads_figure_gen() -> None:
    # The missing-HANDOFF backstop must route through figure_gen, not skip it.
    assert runner._FORWARD_TARGET["results_validator"] == "figure_gen"
    assert runner._FORWARD_TARGET["figure_gen"] == "paper_writer"


def test_pipeline_order_places_figure_gen_between_validator_and_writer() -> None:
    order = cp_manager.PIPELINE_ORDER
    assert "figure_gen" in order
    assert order.index("results_validator") < order.index("figure_gen") < order.index("paper_writer")


def test_figure_gen_opens_no_checkpoint() -> None:
    # figure_gen rides inside the CP4(results_validator) -> CP5(peer_reviewer)
    # leg without a gate of its own — the "five checkpoints" invariant holds.
    assert cp_manager.stage_for_agent("figure_gen") is None
    assert "figure_gen" not in cp_manager.CHECKPOINT_POLICY


def test_figure_gen_is_orchestratable_in_both_sets() -> None:
    assert "figure_gen" in orchestration.ORCHESTRATABLE          # runtime source of truth
    assert "figure_gen" in cp_manager.ORCHESTRATABLE_AGENTS      # console mirror


# ---------------------------------------------------------------------------
# Mock smoke: the chain really routes through figure_gen
# ---------------------------------------------------------------------------

@pytest.fixture()
def isolated_run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from autoscientist.runtime.config import load_config as _load, reset_for_tests

    monkeypatch.setenv("AUTOSCIENTIST_DB_PATH", str(tmp_path / "test.db"))
    reset_for_tests()
    cfg = _load(reload=True)

    runs_dir = tmp_path / "runs"
    projects_dir = tmp_path / "projects"
    runs_dir.mkdir()
    projects_dir.mkdir()
    cfg.default.setdefault("paths", {})["runs_dir"] = str(runs_dir)
    cfg.default["paths"]["projects_dir"] = str(projects_dir)

    for name in ("results_validator", "figure_gen", "paper_writer", "peer_reviewer"):
        cfg.models["agents"][name]["model"] = "mock_stub"

    yield cfg
    reset_for_tests()


def test_chain_routes_results_validator_through_figure_gen(isolated_run_env) -> None:
    """Starting at results_validator with the mock provider, the chain must visit
    figure_gen on its way to paper_writer (proves the rewired mock + topology),
    and figure_gen's plot script lands in the sandbox via the file safety-net."""
    from autoscientist.runtime.runner import run
    from autoscientist.state.db import open_db

    cfg = isolated_run_env
    project_id = "figproj"
    run_id = run(
        starting_agent="results_validator",
        project_id=project_id,
        initial_payload=json.dumps({"plan": {}, "results": {}, "validator_summary": {}}),
        enable_checkpoints=False,
        max_handoffs=6,
        cfg=cfg,
    )

    conn = open_db(cfg.db_path())
    try:
        rows = conn.execute(
            "SELECT DISTINCT agent_name FROM messages "
            "WHERE run_id=? AND role='assistant'",
            (run_id,),
        ).fetchall()
    finally:
        conn.close()
    agents_seen = {r[0] for r in rows}
    # The whole point: figure_gen sits on the path between the two.
    assert {"results_validator", "figure_gen", "paper_writer"}.issubset(agents_seen), agents_seen

    # The mock figure_gen fixture emits a plot script via the files: [...] array;
    # the runner's safety-net persists it to the sandbox.
    projects_root = cfg.root / cfg.default["paths"]["projects_dir"]
    script = projects_root / project_id / "sandbox" / "scripts" / "generate_figures.py"
    assert script.exists(), "figure_gen's plot script was not persisted to the sandbox"


def test_figure_gen_thin_payload_rebuilt_from_sandbox(isolated_run_env) -> None:
    """A thin inbound to figure_gen (placeholder plan / empty results) is rebuilt
    from the materialised result JSON on disk — the runner logs the
    reconstruction so figure_gen always plots real numbers."""
    from autoscientist.runtime.runner import run

    cfg = isolated_run_env
    project_id = "figrebuild"
    projects_root = cfg.root / cfg.default["paths"]["projects_dir"]
    runs = projects_root / project_id / "sandbox" / "runs" / "validator_run"
    runs.mkdir(parents=True)
    (runs / "e1_summary.json").write_text(json.dumps({
        "experiment": "E1", "n_trials": 7,
        "terrain_summaries": [{"terrain": "t", "mean_qb": 0.5}],
    }))

    run_id = run(
        starting_agent="figure_gen",
        project_id=project_id,
        initial_payload=json.dumps({"plan": "x", "results": {}}),  # thin
        enable_checkpoints=False,
        max_handoffs=4,
        cfg=cfg,
    )

    log_path = cfg.runs_dir() / run_id / "logs" / "run.jsonl"
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        e.get("event") == "run.figure_gen_payload_reconstructed" for e in events
    ), "expected figure_gen thin-payload reconstruction to fire"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
