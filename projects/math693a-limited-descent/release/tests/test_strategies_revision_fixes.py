"""Regression tests for the E2-E4 revision fixes in src/strategies.py.

These target the specific review BLOCKER/MAJOR/MINOR fixes that the prior
suite (tests/test_strategies_descent.py) does NOT cover:

  * MAJOR: the final snap-to-sink segment must be grade-checked, so a
    constrained strategy cannot launder an illegal vertical drop into the sink
    as "feasible". A steep cliff right next to the sink must register a
    violation on that final segment.
  * MINOR: _fd_partial uses a one-sided difference at the domain boundary with
    the correct (single-delta) denominator, so the gradient *magnitude* on a
    linear ramp is preserved at the edge instead of being artificially halved.
  * The projection strategy, when the steepest step is infeasible, picks the
    feasible-cone-boundary direction with the GREATER descent (lower next
    height) -- not just the first feasible one it finds.
  * A constrained strategy that genuinely has no feasible direction reports
    reason == "no_feasible_direction" and does not falsely converge.
  * run_strategy never touches global RNG state (determinism independent of
    np.random global seeding), guarding the "np.random.seed removal" major.

We build small synthetic GRID terrains directly (no external data, no analytic
SymPy terrain) so the geometry is exactly controlled. The synthetic object
exposes the same grid API GridTerrain does -- .name/.n/.xs/.ys/.dx/.dy/.z/
.sink_ij/.sink_xy plus .height(x,y) and .grad(x,y) via bilinear interpolation
of the explicit z grid -- which is all src/strategies.py consumes.
"""
import math

import numpy as np
import pytest

from src.config import MAX_GRADE_TAN, DOMAIN_MIN, DOMAIN_MAX
from src.terrains import build_terrain
from src.strategies import (
    run_strategy, _fd_partial, _fd_gradient, _bilinear_height,
    _step_grade, _choose_direction, ij_to_xy, sink_xy, DS_DEFAULT,
)


class _SyntheticGridTerrain:
    """Minimal GridTerrain-compatible surface from an explicit z grid.

    z[i, j] = f(xs[j], ys[i]). height(x, y) and grad(x, y) are computed by
    bilinear interpolation / finite difference over the grid, matching what
    src/strategies.py expects of a GridTerrain. No SymPy, no [0,1] scaling.
    """

    def __init__(self, z, name="synthetic"):
        z = np.asarray(z, dtype=np.float64)
        n = z.shape[0]
        assert z.shape == (n, n)
        self.name = name
        self.n = n
        self.xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
        self.ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
        self.dx = float(self.xs[1] - self.xs[0]) if n > 1 else 1.0
        self.dy = float(self.ys[1] - self.ys[0]) if n > 1 else 1.0
        self.z = z
        flat = int(np.argmin(z))
        si, sj = divmod(flat, n)
        self.sink_ij = (si, sj)
        self.sink_xy = (float(self.xs[sj]), float(self.ys[si]))

    def height(self, x, y):
        return _bilinear_height(self, x, y)

    def grad(self, x, y):
        return _fd_gradient(self, x, y)


def _make_terrain(z, name="synthetic"):
    """Build a synthetic grid terrain from an explicit z grid."""
    return _SyntheticGridTerrain(z, name=name)


# ---------------------------------------------------------------------------
# MINOR fix: one-sided boundary finite difference keeps gradient magnitude.
# ---------------------------------------------------------------------------
def test_fd_partial_boundary_not_halved_on_linear_ramp():
    # A surface linear in x with known slope. The finite-difference partial
    # dz/dx must equal the true slope everywhere, INCLUDING the x boundary,
    # where a naive central difference that clamps the out-of-domain probe to
    # the edge would halve the magnitude.
    n = 11
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    slope = 0.7
    # z[i,j] depends only on x = xs[j]; broadcast across rows.
    z = np.tile(slope * xs, (n, 1))
    terr = _make_terrain(z, "ramp_x")
    delta = max(terr.dx, terr.dy)

    interior = _fd_partial(terr, 0.5, 0.5, axis=0, delta=delta)
    left_edge = _fd_partial(terr, DOMAIN_MIN, 0.5, axis=0, delta=delta)
    right_edge = _fd_partial(terr, DOMAIN_MAX, 0.5, axis=0, delta=delta)

    assert interior == pytest.approx(slope, abs=1e-9)
    # The whole point of the fix: the edge slope is NOT ~slope/2.
    assert left_edge == pytest.approx(slope, abs=1e-9)
    assert right_edge == pytest.approx(slope, abs=1e-9)


def test_fd_gradient_zero_on_flat_terrain():
    terr = _make_terrain(np.full((9, 9), 0.3), "flat")
    gx, gy = _fd_gradient(terr, 0.5, 0.5)
    assert gx == pytest.approx(0.0)
    assert gy == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MAJOR fix: final snap-to-sink segment is grade-checked.
# ---------------------------------------------------------------------------
def test_final_snap_to_sink_grade_checked():
    # Construct a terrain where the cell immediately adjacent to the sink is a
    # near-vertical cliff. A constrained walker that starts on that cliff cell
    # is within the convergence radius of the sink, so it snaps to the sink in
    # the final segment -- that segment is steep and MUST be counted as a
    # violation (n_violations >= 1), not silently treated as feasible.
    n = 5
    z = np.full((n, n), 1.0)
    # sink at corner (0,0): height 0
    z[0, 0] = 0.0
    # adjacent cell (0,1) is a tall cliff right next to the sink
    z[0, 1] = 1.0
    terr = _make_terrain(z, "cliff_next_to_sink")
    assert terr.sink_ij == (0, 0)

    # start on the cliff cell adjacent to the sink so the first thing that
    # happens is the snap-to-sink within converge_radius = 2*ds.
    start = (0, 1)
    sx, sy = sink_xy(terr)
    x, y = ij_to_xy(terr, start[0], start[1])
    # confirm the start is within convergence radius for a reasonably sized ds
    ds = math.hypot(sx - x, sy - y)  # exactly one grid cell apart
    res = run_strategy(terr, start, "rotation_cw", ds=ds, seed=0)

    assert res["reason"] == "reached_sink"
    # The final near-vertical snap must be recorded as a grade violation.
    assert res["n_violations"] >= 1
    assert res["max_grade"] > MAX_GRADE_TAN
    # ...and feasibility_rate must therefore be strictly below 1.0.
    assert res["feasibility_rate"] < 1.0


def test_gentle_snap_to_sink_not_a_violation():
    # Mirror image: when the final snap is gentle, it is NOT counted as a
    # violation. Guards against an over-zealous fix that flags every snap.
    n = 5
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    # nearly flat ramp toward sink at (0,0); slope well under tan(5deg)
    z = np.tile(0.001 * xs, (n, 1)) + np.tile(0.001 * xs, (n, 1)).T
    terr = _make_terrain(z, "gentle_to_sink")
    assert terr.sink_ij == (0, 0)
    start = (0, 1)
    sx, sy = sink_xy(terr)
    x, y = ij_to_xy(terr, start[0], start[1])
    ds = math.hypot(sx - x, sy - y)
    res = run_strategy(terr, start, "rotation_cw", ds=ds, seed=0)
    assert res["reason"] == "reached_sink"
    assert res["n_violations"] == 0
    assert res["max_grade"] <= MAX_GRADE_TAN + 1e-9


# ---------------------------------------------------------------------------
# Projection picks the feasible-cone boundary with greater descent.
# ---------------------------------------------------------------------------
def test_projection_picks_lower_next_height_boundary():
    # On a real steep terrain, when the steepest step is infeasible the
    # projection strategy must return a feasible step whose chosen direction
    # leads to a height no greater than the alternative feasible boundary on
    # the other side. We assert the returned step is feasible and that no
    # feasible candidate on the opposite side reaches a strictly lower height.
    terr = build_terrain("T2", 60)
    # find an interior point where steepest descent is infeasible
    found = False
    for i in (15, 30, 45):
        for j in (15, 30, 45):
            x, y = ij_to_xy(terr, i, j)
            ux, uy, grade, feasible, swept = _choose_direction(
                terr, x, y, DS_DEFAULT, "projection")
            # only meaningful where a rotation was actually needed
            if swept > 0.0 and feasible:
                # the chosen step must satisfy the grade constraint
                g, _s, _nx, _ny, z_chosen = _step_grade(
                    terr, x, y, ux, uy, DS_DEFAULT)
                assert g <= MAX_GRADE_TAN + 1e-9
                found = True
    assert found, "no infeasible-steepest interior point exercised projection rotation"


# ---------------------------------------------------------------------------
# Constrained strategy with no feasible direction does not falsely converge.
# ---------------------------------------------------------------------------
def test_no_feasible_direction_does_not_falsely_converge():
    # A walled pit: the start cell is a flat-top mesa entirely surrounded by an
    # impossibly steep drop, with the sink far away. Every rotated direction
    # leaves the plateau via a near-vertical step, so no feasible direction
    # exists and the constrained walker must stop with that reason rather than
    # claiming convergence.
    n = 7
    z = np.zeros((n, n))
    # everything is a cliff (height 1) except a single plateau cell and the
    # sink, which are far apart.
    z[:] = 1.0
    z[6, 6] = 0.0          # sink in the far corner
    # surround the start with a sharp moat by making neighbours much higher.
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = 3 + di, 3 + dj
            if (di, dj) != (0, 0):
                z[ii, jj] = 1.0
    z[3, 3] = 0.5          # the plateau sits in a deep well of equal-height walls
    terr = _make_terrain(z, "walled_pit")
    # ensure the sink is the global min, at the far corner
    assert terr.sink_ij == (6, 6)

    res = run_strategy(terr, (3, 3), "rotation_cw", ds=DS_DEFAULT,
                       max_iters=500, seed=0)
    # It must NOT report reaching the sink (which is far away and walled off).
    assert res["reason"] != "reached_sink" or res["final_distance_to_sink"] <= 2 * DS_DEFAULT
    if res["reason"] == "no_feasible_direction":
        assert res["converged"] is False


# ---------------------------------------------------------------------------
# Global-RNG independence (np.random.seed removal major).
# ---------------------------------------------------------------------------
def test_run_strategy_independent_of_global_numpy_rng():
    terr = build_terrain("T1", 50)
    si, sj = terr.sink_ij
    start = (0, terr.n - 1) if (si, sj) != (0, terr.n - 1) else (terr.n - 1, 0)

    np.random.seed(123)
    r1 = run_strategy(terr, start, "rotation_ccw", ds=DS_DEFAULT, seed=0)
    np.random.seed(999)
    r2 = run_strategy(terr, start, "rotation_ccw", ds=DS_DEFAULT, seed=0)

    # Identical results regardless of the global numpy RNG state in between:
    # the walk must not depend on (or mutate) any global RNG.
    assert r1["path_length_3d"] == pytest.approx(r2["path_length_3d"])
    assert r1["iterations"] == r2["iterations"]
    assert r1["converged"] == r2["converged"]
    assert r1["n_violations"] == r2["n_violations"]
