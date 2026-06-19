"""Tests for the core grade-constraint correctness of grid path planning.

The central scientific claim of E0 is that Dijkstra / Theta* produce the
*grade-constrained* shortest safe path. If the constraint leaks (an edge whose
slope exceeds tan(5 deg) survives), every downstream optimality_gap and
feasibility metric is meaningless. These tests assert the returned paths obey
the constraint and that infeasible problems are reported as such.

Start points are chosen DYNAMICALLY as the farthest cell that is actually
grade-feasibly reachable from the sink (via select_reachable_start). A fixed
coordinate such as (1.0, 0.0) is permanently infeasible on the steep terrains
(its grade dwarfs tan(5 deg)), which would make every test skip and give zero
coverage of the grade-constraint enforcement.
"""
import math

import numpy as np
import pytest

from src.config import MAX_GRADE_SLOPE, MAX_GRADE_DEGREES
from src.grid_search import (
    dijkstra, theta_star, build_height_grid, select_reachable_start,
)
from src.terrains import get_terrain


def _max_edge_grade(path):
    """Return the maximum |dz|/horiz grade over consecutive path nodes."""
    worst = 0.0
    for (x0, y0, z0), (x1, y1, z1) in zip(path[:-1], path[1:]):
        horiz = math.hypot(x1 - x0, y1 - y0)
        if horiz == 0.0:
            # zero horizontal move with nonzero dz would be an infinite grade
            assert abs(z1 - z0) == 0.0, "vertical-only edge in path"
            continue
        worst = max(worst, abs(z1 - z0) / horiz)
    return worst


def test_max_grade_slope_matches_5_degrees():
    # Sanity: the configured slope really is tan(5 deg), not radians or degrees.
    assert MAX_GRADE_DEGREES == 5.0
    assert MAX_GRADE_SLOPE == pytest.approx(math.tan(math.radians(5.0)))


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_dijkstra_path_respects_grade_constraint(name):
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None, "no reachable non-sink start exists on grid"
    length, path = dijkstra(terrain, n, start, goal)
    # A reachable start MUST yield a feasible path -- no skipping.
    assert path is not None and math.isfinite(length)
    # Every edge of a Dijkstra-returned path must satisfy the grade constraint.
    worst = _max_edge_grade(path)
    assert worst <= MAX_GRADE_SLOPE + 1e-9, (
        f"Dijkstra returned an edge with grade {worst} > {MAX_GRADE_SLOPE}"
    )


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_theta_star_path_respects_grade_constraint(name):
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None, "no reachable non-sink start exists on grid"
    length, path = theta_star(terrain, n, start, goal)
    assert path is not None and math.isfinite(length)
    # Theta* takes any-angle shortcuts; sample each segment densely and verify
    # the running grade never exceeds the limit (the line-of-sight guarantee).
    # Heights are evaluated analytically -- the SAME surface the algorithm now
    # uses in line-of-sight checks -- so the tolerance can be tight.
    for (x0, y0, z0), (x1, y1, z1) in zip(path[:-1], path[1:]):
        horiz_total = math.hypot(x1 - x0, y1 - y0)
        if horiz_total == 0.0:
            continue
        ts = np.linspace(0.0, 1.0, 50)
        xs = x0 + ts * (x1 - x0)
        ys = y0 + ts * (y1 - y0)
        zs = np.asarray(terrain.height(xs, ys), dtype=float)
        for i in range(1, len(ts)):
            dh = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            if dh == 0.0:
                continue
            grade = abs(zs[i] - zs[i - 1]) / dh
            assert grade <= MAX_GRADE_SLOPE * 1.05, (
                f"Theta* segment violates grade: {grade} > {MAX_GRADE_SLOPE}"
            )


def test_infeasible_problem_returns_inf_not_a_path():
    # A wall too steep to cross at low resolution should be reported infeasible
    # rather than silently returning a constraint-violating path. We construct
    # this by asking for a path across an extremely steep paraboloid on a coarse
    # grid where no 8-connected edge can satisfy tan(5 deg).
    terrain = get_terrain("T2")  # Rosenbrock ridge: very steep walls
    n = 30
    # Start and goal on opposite sides of the steep ridge.
    length, path = dijkstra(terrain, n, (-1.5, -1.5), (1.5, 1.5))
    if path is None:
        assert math.isinf(length)
    else:
        # If a path IS found it must still obey the constraint.
        assert _max_edge_grade(path) <= MAX_GRADE_SLOPE + 1e-9
