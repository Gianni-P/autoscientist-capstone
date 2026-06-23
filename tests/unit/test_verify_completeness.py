"""Unit tests for the experiment-completeness verify module."""

from __future__ import annotations

from autoscientist.verify import run_all
from autoscientist.verify.completeness import (
    check_baselines_present,
    check_experiment_completeness,
    run_completeness,
)


def _plan(*exp_ids, baselines=None):
    return {
        "experiments": [{"id": e} for e in exp_ids],
        "baselines": baselines or [],
    }


def test_all_declared_experiments_present_passes():
    plan = _plan("E1", "E2", "E3")
    results = {
        "runs/e1_summary.json": {"experiment": "E1"},
        "runs/e2_summary.json": {"experiment": "E2"},
        "runs/e3_summary.json": {"experiment": "E3"},
    }
    v = check_experiment_completeness(plan, results)
    assert v.status == "pass"
    assert v.category == "completeness"


def test_missing_experiment_fails_with_ids():
    plan = _plan("E1", "E2", "E3", "E4", "E5")
    results = {"runs/e1_summary.json": {"experiment": "E1"}}
    v = check_experiment_completeness(plan, results)
    assert v.status == "fail"
    assert v.severity == "fail"
    assert set(v.evidence["missing"]) == {"E2", "E3", "E4", "E5"}


def test_experiment_present_from_artifact_key_without_field():
    plan = _plan("E2")
    results = {"prod_s0_e2/e2_summary.json": {"terrain_summaries": []}}
    v = check_experiment_completeness(plan, results)
    assert v.status == "pass"


def test_no_experiments_declared_skipped():
    v = check_experiment_completeness({"experiments": []}, {"x": {}})
    assert v.status == "skipped"


def test_no_results_yet_skipped_not_failed():
    v = check_experiment_completeness(_plan("E1"), None)
    assert v.status == "skipped"


def test_plan_unwrapped_from_payload_envelope():
    payload_plan = {"plan": _plan("E1", "E2")}
    results = {"e1_summary.json": {"experiment": "E1"}}
    v = check_experiment_completeness(payload_plan, results)
    assert v.status == "fail"
    assert v.evidence["missing"] == ["E2"]


def test_baselines_present_when_comparison_markers_exist():
    plan = _plan("E1", baselines=[{"name": "Dijkstra ground truth"}])
    results = {"e1_summary.json": {"mean_optimality_gap": 0.1}}
    v = check_baselines_present(plan, results)
    assert v.status == "pass"
    assert {"gap", "optimality"} & set(v.evidence["markers_found"])


def test_baselines_absent_needs_human():
    plan = _plan("E1", baselines=[{"name": "Dijkstra"}])
    results = {"e1_summary.json": {"wall_seconds": 0.3, "n_trials": 100}}
    v = check_baselines_present(plan, results)
    assert v.status == "needs_human"
    assert v.evidence["declared_baselines"] == ["Dijkstra"]


def test_no_baselines_declared_skipped():
    v = check_baselines_present(_plan("E1"), {"x": {}})
    assert v.status == "skipped"


def test_run_completeness_returns_two_verdicts():
    state = {
        "plan": _plan("E1", baselines=[{"name": "b"}]),
        "results": {"e1_summary.json": {"experiment": "E1", "gap": 1.0}},
    }
    vs = run_completeness(state)
    assert {v.check_id for v in vs} == {"experiment_completeness", "baselines_present"}


def test_run_all_includes_completeness_and_blocks_on_missing_experiment():
    state = {
        "plan": _plan("E1", "E2"),
        "results": {"e1_summary.json": {"experiment": "E1"}},
    }
    rep = run_all(state, domain="medical_imaging")
    assert rep.outcome == "block"
    miss = next(v for v in rep.verdicts if v.check_id == "experiment_completeness")
    assert miss.status == "fail"
    assert miss.evidence["missing"] == ["E2"]
