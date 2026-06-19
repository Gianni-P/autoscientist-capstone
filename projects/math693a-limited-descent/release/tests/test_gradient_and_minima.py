"""Tests for analytic-gradient correctness and unique-minimum verification.

These guard the two analytic prerequisites the plan names explicitly:
  * gradient_validation = finite_difference: the SymPy-derived analytic gradient
    must agree with central finite differences. A sign error or chain-rule slip
    here silently corrupts every descent method.
  * unique-minimum: each terrain's designed sink must be the true GLOBAL minimum
    on the domain and beat the second-lowest basin by the configured depth
    margin. If the designed sink is NOT the global minimum (e.g. a saddle term
    drags the true minimum to a domain corner), the goal endpoint used as
    ground truth is wrong and every optimality_gap is computed against the wrong
    target. These tests assert that prerequisite for every terrain.
"""
import numpy as np
import pytest

from src.config import GRAD_FD_TOL, UNIQUE_MIN_DEPTH_MARGIN
from src.terrains import get_terrain, all_terrains
from src.validation import verify_gradient, verify_unique_minimum


@pytest.mark.parametrize("name", ["T1", "T2", "T3", "T4", "T5"])
def test_analytic_gradient_matches_finite_difference(name):
    terrain = get_terrain(name)
    res = verify_gradient(terrain, n_points=30, rng=np.random.default_rng(0))
    assert res["max_abs_error"] <= GRAD_FD_TOL, (
        f"{name}: analytic grad disagrees with FD by {res['max_abs_error']}"
    )
    assert res["passed"] is True


def test_gradient_check_catches_a_wrong_gradient():
    # Negative control: a deliberately broken gradient must FAIL the check,
    # proving the test is not vacuously passing.
    terrain = get_terrain("T1")

    class BrokenTerrain:
        name = "broken"
        height = terrain.height

        def grad(self, x, y):
            gx, gy = terrain.grad(x, y)
            # Flip the sign of dz/dx -> should be caught by FD comparison.
            return -gx, gy

    res = verify_gradient(BrokenTerrain(), n_points=20,
                          rng=np.random.default_rng(1))
    assert res["passed"] is False
    assert res["max_abs_error"] > GRAD_FD_TOL


@pytest.mark.parametrize("name", ["T1", "T2", "T3", "T4", "T5"])
def test_designed_sink_is_global_minimum(name):
    # Prerequisite (b) of the E0 gate: the designed sink MUST be the true global
    # minimum of f+g on the domain. A terrain whose unbounded/saddle base term
    # pushes the real minimum to a domain edge violates this and makes the goal
    # endpoint (and thus optimality_gap ground truth) incorrect.
    terrain = get_terrain(name)
    res = verify_unique_minimum(terrain, n=200)
    loc = res["global_min_loc"]
    sink = terrain.sink
    assert abs(loc[0] - sink[0]) <= 0.1 and abs(loc[1] - sink[1]) <= 0.1, (
        f"{name}: grid global-min {tuple(round(v,3) for v in loc)} is not at "
        f"the designed sink {sink}; the well does not dominate the base term "
        f"over the whole domain, so this terrain's reference optimum is invalid."
    )


@pytest.mark.parametrize("name", ["T1", "T2", "T3", "T4", "T5"])
def test_unique_minimum_property_holds(name):
    # The global minimum must beat the second-lowest basin by the required
    # depth margin -- otherwise "unique safe shortest path" has no well-defined
    # target. margin == 0 means a competing basin is as deep as the sink.
    terrain = get_terrain(name)
    res = verify_unique_minimum(terrain, n=200)
    assert res["depth_margin"] >= UNIQUE_MIN_DEPTH_MARGIN, (
        f"{name}: second basin within {res['depth_margin']:.3f} of the global "
        f"min (need >= {UNIQUE_MIN_DEPTH_MARGIN}); minimum is not uniquely deep."
    )
    assert res["passed"] is True


def test_all_terrains_returns_five_distinct_names():
    ts = all_terrains()
    assert len(ts) == 5
    assert sorted(t.name for t in ts) == ["T1", "T2", "T3", "T4", "T5"]
