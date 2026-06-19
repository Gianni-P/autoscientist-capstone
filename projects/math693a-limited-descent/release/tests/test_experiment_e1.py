"""End-to-end E1 tests on a small grid.

Targets pitfalls: Quantisation Bias sign/definition, the recorded metrics are
self-consistent, internal-validity checks actually fire, and the whole run is
seed-deterministic.
"""
import json
import os

import pytest

from src.experiment_e1 import run_e1


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    # run inside a tmp cwd so runs/ does not pollute the repo
    d = tmp_path_factory.mktemp("e1run")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        summary = run_e1(run_id="t1", seed=0, grid_n=40)
    finally:
        os.chdir(cwd)
    return d, summary


def test_summary_structure(small_run):
    _, summary = small_run
    assert summary["experiment"] == "E1"
    assert summary["grid_n"] == 40
    assert summary["n_trials"] >= 1
    assert "internal_validity_passed" in summary


def test_jsonl_written_and_parseable(small_run):
    d, summary = small_run
    path = os.path.join(d, "runs", "t1", "e1_trials.jsonl")
    assert os.path.exists(path)
    with open(path) as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    assert len(lines) == summary["n_trials"]


def test_quantisation_bias_definition(small_run):
    """QB = (raw - theta*)/theta* must be non-negative (theta* never longer)."""
    d, _ = small_run
    path = os.path.join(d, "runs", "t1", "e1_trials.jsonl")
    with open(path) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    assert recs, "no trial records produced"
    for r in recs:
        raw, theta = r["raw_dijkstra_length"], r["theta_star_length"]
        assert theta <= raw + 1e-9, "theta* longer than raw Dijkstra"
        assert r["quantisation_bias"] >= -1e-9, "QB went negative"
        if theta > 0:
            expected = (raw - theta) / theta
            assert r["quantisation_bias"] == pytest.approx(expected, rel=1e-9)
        # check_a flag must agree with the actual inequality
        assert r["check_theta_le_raw"] == (theta <= raw + 1e-9)


def test_theta_geodesic_lower_bound(small_run):
    """Theta* length must be >= the straight-line 3-D geodesic to the sink."""
    d, _ = small_run
    path = os.path.join(d, "runs", "t1", "e1_trials.jsonl")
    with open(path) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    for r in recs:
        assert r["theta_star_length"] >= r["geodesic_3d"] - 1e-9
        assert r["theta_geodesic_deviation"] >= -1e-9


def test_run_is_seed_deterministic(tmp_path):
    """Two runs with the same seed/grid yield identical trial metrics."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        s1 = run_e1(run_id="a", seed=0, grid_n=40)
        s2 = run_e1(run_id="b", seed=0, grid_n=40)
    finally:
        os.chdir(cwd)
    assert s1["n_trials"] == s2["n_trials"]

    def load(rid):
        with open(os.path.join(tmp_path, "runs", rid, "e1_trials.jsonl")) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    r1, r2 = load("a"), load("b")
    assert [x["start_ij"] for x in r1] == [x["start_ij"] for x in r2]
    for a, b in zip(r1, r2):
        assert a["raw_dijkstra_length"] == pytest.approx(b["raw_dijkstra_length"])
        assert a["theta_star_length"] == pytest.approx(b["theta_star_length"])
        assert a["quantisation_bias"] == pytest.approx(b["quantisation_bias"])
