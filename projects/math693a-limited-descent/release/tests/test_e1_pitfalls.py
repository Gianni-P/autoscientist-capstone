"""Regression tests for the specific pitfalls the E1 plan / code review flagged.

These complement the existing suites (test_graph, test_terrains_startpoints,
test_experiment_e1) by locking down the *exact* failure modes named in the
methodology plan and the code-review findings:

  * finding_3 -- T3 (sinusoidal_valley) must be *passable*: >0 reachable starts
    and, end-to-end, >0 E1 trials. Its previous impassable form (0 reachable
    cells) silently produced zero trials and an aborted terrain, which a
    structural test must catch.
  * Silent terrain abort -- E1 must NOT mark any configured terrain as
    aborted / 0-trial. A terrain that contributes nothing to the comparison is
    a real defect, not a benign skip.
  * finding_4 -- the seed must genuinely drive start-point selection (no dead
    np.random.seed): different seeds must be *able* to change the selection,
    and the metrics recorded under a given seed must be reproducible.
  * Grade-constraint integrity along the *reference* paths E1 actually records:
    every consecutive (i,j) pair of a recorded Dijkstra path must be a feasible
    edge -- the ground-truth path cannot itself violate the 5-degree limit.
"""
import json
import os

import numpy as np
import pytest

from src.config import TERRAINS
from src.terrains import build_terrain, list_terrains
from src.graph import (
    dijkstra_grade_constrained,
    reconstruct_path,
    edge_feasible,
)
from src.startpoints import select_start_points
from src.experiment_e1 import run_e1


GRID_N_SMALL = 40


def _reachable_mask(t):
    sink_dist, _ = dijkstra_grade_constrained(t, t.sink_ij)
    return np.isfinite(sink_dist)


# ---------------------------------------------------------------------------
# finding_3: T3 (sinusoidal_valley) passability
# ---------------------------------------------------------------------------
def test_t3_sinusoidal_valley_is_passable():
    """The repaired T3 (sinusoidal_valley) must yield a non-empty reachable set.

    Guards against a regression to the old impassable surface (every edge grade
    >> tan(5 deg), 0 reachable cells, 0 start points).
    """
    t = build_terrain("T3", GRID_N_SMALL)
    mask = _reachable_mask(t)
    n_reachable = int(mask.sum())
    assert n_reachable > 1, (
        f"T3 has only {n_reachable} reachable cell(s) -- terrain is "
        "(near-)impassable; the grade constraint excludes essentially every edge"
    )
    starts = select_start_points(t, mask, seed=0)
    assert len(starts) > 0, (
        "T3 produced 0 stratified start points -- no E1 trial can be run on it"
    )


@pytest.mark.parametrize("name", list_terrains())
def test_every_terrain_has_reachable_starts(name):
    """No analytic terrain may be structurally impassable.

    Each configured terrain must admit at least one stratified start point so
    that the E1 comparison covers all of them rather than silently dropping any.
    """
    t = build_terrain(name, GRID_N_SMALL)
    mask = _reachable_mask(t)
    starts = select_start_points(t, mask, seed=0)
    assert len(starts) > 0, f"{name} has no reachable start points"


# ---------------------------------------------------------------------------
# Silent terrain abort: end-to-end E1 must produce trials for every terrain
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def e1_summary(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1_pitfalls")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        summary = run_e1(run_id="p", seed=0, grid_n=GRID_N_SMALL)
    finally:
        os.chdir(cwd)
    return d, summary


def test_no_terrain_aborted(e1_summary):
    """Every terrain in the run must be non-aborted with >=1 trial."""
    _, summary = e1_summary
    by_name = {s["terrain"]: s for s in summary["terrain_summaries"]}
    # all configured terrains present
    for name in TERRAINS:
        assert name in by_name, f"{name} missing from terrain_summaries"
    for name, s in by_name.items():
        assert not s.get("aborted", False), f"terrain {name} was aborted"
        assert s["n_starts"] >= 1, f"terrain {name} produced 0 trials"


def test_t3_contributes_trials(e1_summary):
    """End-to-end, T3 (sinusoidal_valley) must contribute at least one trial."""
    d, _ = e1_summary
    path = os.path.join(d, "runs", "p", "e1_trials.jsonl")
    with open(path) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    sv = [r for r in recs if r["terrain"] == "T3"]
    assert len(sv) > 0, "no E1 trial recorded for T3"


# ---------------------------------------------------------------------------
# finding_4: the seed genuinely drives selection (no dead seed)
# ---------------------------------------------------------------------------
def test_seed_actually_changes_selection():
    """Different seeds must be able to produce different start selections.

    A dead seed (selection independent of the seed argument) would make every
    seed identical; this asserts the seed is a live input on a terrain with a
    large enough reachable candidate pool to admit variation.
    """
    t = build_terrain("T1", GRID_N_SMALL)
    mask = _reachable_mask(t)
    a = select_start_points(t, mask, seed=0)
    b = select_start_points(t, mask, seed=123)
    assert a != b, (
        "start-point selection identical across two seeds -- seed appears dead"
    )


def test_same_seed_reproducible_metrics(tmp_path):
    """Same seed/grid -> identical recorded path-length metrics (determinism)."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_e1(run_id="x", seed=0, grid_n=GRID_N_SMALL)
        run_e1(run_id="y", seed=0, grid_n=GRID_N_SMALL)
    finally:
        os.chdir(cwd)

    def load(rid):
        with open(os.path.join(tmp_path, "runs", rid, "e1_trials.jsonl")) as fh:
            return [json.loads(l) for l in fh if l.strip()]

    r1, r2 = load("x"), load("y")
    assert len(r1) == len(r2) and len(r1) > 0
    for a, b in zip(r1, r2):
        assert a["terrain"] == b["terrain"]
        assert a["start_ij"] == b["start_ij"]
        assert a["raw_dijkstra_length"] == pytest.approx(b["raw_dijkstra_length"])
        assert a["theta_star_length"] == pytest.approx(b["theta_star_length"])


# ---------------------------------------------------------------------------
# Reference-path grade integrity: the ground-truth path cannot violate grade
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", list_terrains())
def test_reference_path_edges_all_feasible(name):
    """Every edge of a recorded Dijkstra reference path respects the grade.

    The constrained shortest path is the ground truth E1 compares against; if
    any of its edges violated tan(5 deg) the constraint enforcement (or the
    reachability bookkeeping) would be broken.
    """
    t = build_terrain(name, GRID_N_SMALL)
    mask = _reachable_mask(t)
    starts = select_start_points(t, mask, seed=0)
    assert starts, f"{name} produced no starts"
    start = starts[0]
    dist, prev = dijkstra_grade_constrained(t, start)
    path = reconstruct_path(prev, start, t.sink_ij)
    assert path is not None, (
        f"{name}: sink unreachable from a start that is reachable from the sink"
    )
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        assert edge_feasible(t, i0, j0, i1, j1), (
            f"{name}: reference path edge ({i0},{j0})->({i1},{j1}) "
            "violates the max-grade constraint"
        )
