"""Tests for the shared reference (optimum) construction in src.common.

NOTE: these previously passed the removed terrain name 'elliptic_paraboloid' to
prepare_terrain, which now raises KeyError ('have [T1..T5]') -- i.e. they failed
at the wrong layer (a dead string) and so could not pin the real pipeline. They
are updated to the canonical terrain id 'T1'. They now fail (correctly) if and
only if the reference pipeline itself is broken.

Pitfalls targeted:
  * The Theta* reference is meant to be the OPTIMUM (ground truth) and the
    grid Dijkstra path is feasible: theta_len <= raw_len by construction.
  * prepare_terrain must drop starts whose sink is unreachable (no leaky
    inclusion of infeasible reference lengths) yet keep at least one usable
    start (otherwise there is no ground truth at all).
  * Reference build must be deterministic for a fixed seed (start selection).
  * Gap-sign sanity: against a Theta* OPTIMUM a *feasible* continuous path
    should not be materially shorter than the reference. Documented as an xfail
    because the continuous bilinear-grade walk can shortcut the
    grid-edge-constrained optimum (a real caveat for E5's optimality_gap).
"""
import math

import pytest

from src.config import TERRAINS
from src.common import prepare_terrain, RUN_TERRAINS
from src.strategies import run_strategy, DS_DEFAULT

GRID_N = 50
TERRAIN = "T1"


def test_run_terrains_match_config_count():
    assert len(RUN_TERRAINS) == len(TERRAINS)


def test_prepare_terrain_stores_consistent_lengths():
    setup = prepare_terrain(TERRAIN, GRID_N, seed=42)
    assert len(setup.starts) > 0, (
        "prepare_terrain produced no usable starts -- the reference optimum is "
        "empty, so there is no ground truth to compare against."
    )
    for sp in setup.starts:
        assert sp in setup.theta_len
        assert sp in setup.raw_len
        assert math.isfinite(setup.theta_len[sp]) and setup.theta_len[sp] > 0
        assert math.isfinite(setup.raw_len[sp]) and setup.raw_len[sp] > 0
        # Theta* smoothing can only shorten or equal the raw Dijkstra path.
        assert setup.theta_len[sp] <= setup.raw_len[sp] + 1e-9


def test_converged_constrained_paths_are_feasible():
    # Guaranteed invariant: a constrained strategy never records a grade
    # violation (this is the property the optimality-gap interpretation relies
    # on, independent of whether the walk converges).
    setup = prepare_terrain(TERRAIN, GRID_N, seed=42)
    saw_steps = False
    for sp in setup.starts:
        res = run_strategy(setup.terrain, sp, "gradient_projection",
                           ds=DS_DEFAULT, seed=0)
        if res["n_steps"] > 0:
            saw_steps = True
            assert res["n_violations"] == 0
    assert saw_steps


@pytest.mark.xfail(reason="continuous bilinear-grade walk can shortcut the "
                          "grid-edge Theta* optimum, yielding negative "
                          "optimality_gap; documented E5 caveat",
                   strict=False)
def test_reference_is_lower_bound_for_feasible_strategy():
    # Ideal property: a feasible converged path is >= the Theta* optimum.
    setup = prepare_terrain(TERRAIN, GRID_N, seed=42)
    checked = 0
    for sp in setup.starts:
        res = run_strategy(setup.terrain, sp, "gradient_projection",
                           ds=DS_DEFAULT, seed=0)
        if res["converged"]:
            assert res["path_length_3d"] >= setup.theta_len[sp] * 0.9
            checked += 1
    assert checked > 0


def test_prepare_terrain_deterministic():
    s1 = prepare_terrain(TERRAIN, GRID_N, seed=42)
    s2 = prepare_terrain(TERRAIN, GRID_N, seed=42)
    assert s1.starts == s2.starts
    for sp in s1.starts:
        assert s1.theta_len[sp] == pytest.approx(s2.theta_len[sp])
        assert s1.raw_len[sp] == pytest.approx(s2.raw_len[sp])
