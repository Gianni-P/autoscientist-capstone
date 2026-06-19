"""Regression tests for the T4 (monkey-saddle-flavoured) interior-minimum fix.

Plan step under test
---------------------
T4's base ``x**2 - y**2`` saddle is made bounded-below by a positive-definite
bowl (``+ 2.0*(x**2 + y**2)`` plus a small ``0.1*(x**2 + y**2)`` term), giving
``3.1*x**2 + 1.1*y**2``, plus a Gaussian well centred at the origin. The unique
global minimum therefore sits at the INTERIOR critical point (0, 0) -- where the
analytic gradient is zero -- instead of on a domain boundary. This is the
mechanism that gives T4 a genuinely 2-D grade-feasible basin around its sink.

These tests lock down the *cause* the current code relies on: an interior
global minimum at (0, 0) over the canonical domain [-2, 2]^2 (NOT a normalised
[0, 1]^2 grid; GridTerrain stores raw scaled heights). A regression to an
unbounded or boundary-pinned surface fails these loudly, independent of grid
resolution.
"""
import math

import numpy as np
import pytest

from src.config import MAX_GRADE_TAN, DOMAIN_MIN, DOMAIN_MAX
from src.terrains import build_terrain, terrain_function


def _global_argmin_xy(f, n=400):
    """Coordinates (x, y) of the analytic global minimum over [-2,2]^2."""
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    X, Y = np.meshgrid(xs, xs)  # X[i,j]=xs[j], Y[i,j]=xs[i]
    raw = np.asarray(f(X, Y), dtype=float)
    i, j = np.unravel_index(int(np.argmin(raw)), raw.shape)
    return float(xs[j]), float(xs[i])


def test_t4_global_min_is_interior_origin():
    """The global minimum over [-2,2]^2 must sit at the interior point (0,0).

    A regression to an unbounded saddle (or a boundary-pinned minimum) moves the
    argmin to the domain edge -- exactly the degeneracy the bowl term removes.
    """
    f = terrain_function("T4")
    x_star, y_star = _global_argmin_xy(f, n=400)
    # one grid cell at n=400 over [-2,2] is ~4/399 ~= 0.01; allow a couple cells.
    assert x_star == pytest.approx(0.0, abs=0.03), (
        f"T4 global-min x = {x_star:.4f}, expected interior 0.0; the bowl "
        "regularisation appears to have regressed (minimum slid to a boundary)"
    )
    assert y_star == pytest.approx(0.0, abs=0.03), (
        f"T4 global-min y = {y_star:.4f}, expected interior 0.0"
    )


def test_t4_origin_is_strictly_below_boundary_edge():
    """f(0,0) must be strictly lower than the y=DOMAIN_MAX boundary edge minimum.

    The quantitative statement that the regularisation pulled the optimum off
    the boundary: the interior value at (0,0) must beat the best value attainable
    along the top edge.
    """
    f = terrain_function("T4")
    origin_val = float(f(0.0, 0.0))
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 400)
    edge_min = float(np.min(f(xs, np.full_like(xs, DOMAIN_MAX))))
    assert origin_val < edge_min - 1e-6, (
        f"f(0,0)={origin_val:.5f} is not strictly below the top-edge minimum "
        f"{edge_min:.5f}; the global minimum is still on the boundary"
    )


def test_t4_sink_index_at_grid_centre():
    """build_terrain must place the T4 sink at the grid cell nearest (0,0).

    The sink_xy must be ~ (0,0) and sink_ij the central cell at any resolution.
    Catches both a regression of the regularisation and a meshgrid transposition
    that would move the recorded sink off the true minimum.
    """
    for n in (41, 151):
        t = build_terrain("T4", n)
        sx, sy = t.sink_xy
        assert sx == pytest.approx(0.0, abs=t.dx + 1e-9)
        assert sy == pytest.approx(0.0, abs=t.dy + 1e-9)
        si, sj = t.sink_ij
        # the sink height must be the grid global minimum
        assert t.z[si, sj] == pytest.approx(t.z.min())


def test_t4_basin_around_origin_is_grade_feasible():
    """A 2-D neighbourhood of the origin sink must be grade-feasible.

    Near a smooth interior minimum the gradient -> 0, so adjacent-cell grades
    fall below tan(5 deg). We require BOTH the x-neighbour and the y-neighbour of
    the sink cell to be feasible steps -- a genuinely 2-D basin, not a 1-D
    sliver confined to a single row or column.
    """
    t = build_terrain("T4", 151)
    si, sj = t.sink_ij
    for (ni, nj), axis in [((si, sj + 1), "x"), ((si + 1, sj), "y")]:
        h = math.hypot((nj - sj) * t.dx, (ni - si) * t.dy)
        dz = abs(t.z[ni, nj] - t.z[si, sj])
        grade = dz / h
        assert grade <= MAX_GRADE_TAN + 1e-12, (
            f"step from sink along {axis} has grade {grade:.5f} > "
            f"tan(5deg)={MAX_GRADE_TAN:.5f}; basin is not grade-feasible in 2-D"
        )
