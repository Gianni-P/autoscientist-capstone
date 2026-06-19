"""Tests for the continuous descent strategies (E2/E3/E4).

Pitfalls targeted:
  * Grade-constraint correctness: constrained strategies must NOT take steps
    that violate the 5-degree max-grade constraint, whereas the unconstrained
    baseline is permitted to.
  * Steepest-descent direction must actually point downhill (sign of gradient).
  * Rotation heuristic must use 1-degree increments bounded by 360 degrees.
  * Seed determinism: run_strategy must be reproducible.
  * Path length must be a positive 3-D length that accounts for height change.
"""
import math

import numpy as np
import pytest

from src.terrains import build_terrain
from src.config import MAX_GRADE_TAN
from src.strategies import (
    run_strategy, STRATEGIES, ROT_INCREMENT_DEG, DS_DEFAULT,
    _steepest_dir, _bilinear_height, _step_grade, ij_to_xy, sink_xy,
)

GRID_N = 60


@pytest.fixture(scope="module")
def bowl():
    # smooth bowl (T1, elliptic paraboloid): gentle, well-behaved gradient field
    return build_terrain("T1", GRID_N)


def _a_start(terrain):
    # a corner start far from the sink
    si, sj = terrain.sink_ij
    i = 0 if si > terrain.n // 2 else terrain.n - 1
    j = 0 if sj > terrain.n // 2 else terrain.n - 1
    return (i, j)


def test_rotation_increment_constant():
    # Pitfall: rotation heuristic must use the planned 1-degree increment.
    assert ROT_INCREMENT_DEG == 1.0


def test_strategy_names_present():
    # The four method keys the drivers depend on must exist.
    for key in ("unconstrained_steepest_descent", "rotation_cw",
                "rotation_ccw", "gradient_projection"):
        assert key in STRATEGIES


def test_steepest_dir_points_downhill(bowl):
    # Pitfall: a sign error in the finite-difference gradient would send the
    # walker uphill. Verify the chosen direction lowers height for a sample.
    si, sj = _a_start(bowl)
    x, y = ij_to_xy(bowl, si, sj)
    ux, uy, gnorm = _steepest_dir(bowl, x, y)
    assert gnorm > 0
    assert abs(math.hypot(ux, uy) - 1.0) < 1e-9
    z0 = _bilinear_height(bowl, x, y)
    z1 = _bilinear_height(bowl, x + ux * 0.01, y + uy * 0.01)
    assert z1 < z0


def test_constrained_steps_respect_grade(bowl):
    # Pitfall: the whole project hinges on "limited descent" actually limiting
    # the grade. Every recorded step of a constrained strategy must be feasible.
    for strat in ("rotation_cw", "rotation_ccw", "gradient_projection"):
        res = run_strategy(bowl, _a_start(bowl), strat, ds=DS_DEFAULT, seed=0)
        # n_violations is the count of steps exceeding the grade. For a
        # constrained method that should be zero whenever it took steps.
        if res["n_steps"] > 0:
            assert res["n_violations"] == 0, (strat, res["max_grade"])
            assert res["feasibility_rate"] == pytest.approx(1.0)


def test_unconstrained_may_violate_on_steep_terrain():
    # Pitfall: if the unconstrained baseline silently honored the constraint,
    # E2 would not be a valid lower bound / violation upper bound. On a steep
    # terrain it should take at least one infeasible step somewhere.
    terr = build_terrain("T2", GRID_N)  # T2 == Rosenbrock ridge (steep)
    any_violation = False
    si, sj = terr.sink_ij
    # try several starts; the ridge is steep so violations are expected.
    for i in (0, terr.n - 1):
        for j in (0, terr.n - 1):
            res = run_strategy(terr, (i, j), "unconstrained_steepest_descent",
                               ds=DS_DEFAULT, seed=0)
            if res["n_violations"] > 0:
                any_violation = True
    assert any_violation, "unconstrained descent never violated the grade"


def test_run_strategy_seed_determinism(bowl):
    # Pitfall: non-deterministic results across identical seeds would invalidate
    # the bootstrap CIs and hypothesis tests.
    start = _a_start(bowl)
    r1 = run_strategy(bowl, start, "rotation_ccw", ds=DS_DEFAULT, seed=0)
    r2 = run_strategy(bowl, start, "rotation_ccw", ds=DS_DEFAULT, seed=0)
    assert r1["path_length_3d"] == r2["path_length_3d"]
    assert r1["converged"] == r2["converged"]
    assert r1["iterations"] == r2["iterations"]


def test_path_length_is_3d(bowl):
    # Pitfall: silently computing planar (2-D) length instead of 3-D would
    # ignore the height penalty. The 3-D step length must be >= the planar one.
    start = _a_start(bowl)
    x, y = ij_to_xy(bowl, start[0], start[1])
    ux, uy, _ = _steepest_dir(bowl, x, y)
    grade, seg3d, nx, ny, z1 = _step_grade(bowl, x, y, ux, uy, DS_DEFAULT)
    assert seg3d >= DS_DEFAULT - 1e-12
    # with a real height change it should be strictly greater
    if abs(z1 - _bilinear_height(bowl, x, y)) > 1e-9:
        assert seg3d > DS_DEFAULT


def test_converged_path_reaches_sink(bowl):
    # On the smooth bowl a feasible descent should converge to the sink.
    res = run_strategy(bowl, _a_start(bowl), "gradient_projection",
                       ds=DS_DEFAULT, seed=0)
    if res["converged"]:
        assert res["final_distance_to_sink"] <= 2.0 * DS_DEFAULT + 1e-9


def test_unknown_strategy_raises(bowl):
    with pytest.raises(ValueError):
        run_strategy(bowl, _a_start(bowl), "not_a_strategy", seed=0)


def test_grade_definition_matches_threshold(bowl):
    # Sanity on the grade metric itself: a perfectly horizontal step has 0 grade.
    x, y = 0.5, 0.5
    # use a flat synthetic by zeroing z is not possible; instead test a step
    # whose endpoints we know via interpolation produce a finite grade.
    grade, seg3d, nx, ny, z1 = _step_grade(bowl, x, y, 1.0, 0.0, DS_DEFAULT)
    assert grade >= 0.0
    assert math.isfinite(grade)
    # threshold constant must equal tan(5 deg)
    assert MAX_GRADE_TAN == pytest.approx(math.tan(math.radians(5.0)))
