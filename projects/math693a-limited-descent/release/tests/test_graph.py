"""Tests for the grade-constrained graph + Dijkstra + Theta* (src/graph.py).

Failure modes targeted:
  * grade/feasibility metric mis-implemented (the 5-degree constraint is the
    whole experiment; an off-by-one in tan() or dz/h would leak infeasible
    edges into the "optimum")
  * Theta* smoothing that LENGTHENS a path (it must never increase 3-D length)
  * Dijkstra reference not actually optimal (a hand-built feasible path can
    never be shorter than the Dijkstra distance)
"""
import math

import numpy as np

from src.config import MAX_GRADE_TAN, MAX_GRADE_DEGREES
from src.terrains import build_terrain
from src.graph import (
    edge_feasible, path_length_3d, dijkstra_grade_constrained,
    reconstruct_path, theta_star_smooth, line_of_sight,
)


def test_max_grade_tan_value():
    assert math.isclose(MAX_GRADE_TAN, math.tan(math.radians(MAX_GRADE_DEGREES)))


def test_edge_feasible_threshold():
    """A step just under tan(5deg) is feasible; well above it is not."""
    t = build_terrain("T1", 50)
    h = t.dx  # horizontal distance of an east-west move
    # feasible: grade just under threshold
    t.z[0, 0] = 0.0
    t.z[0, 1] = (MAX_GRADE_TAN - 1e-4) * h
    assert bool(edge_feasible(t, 0, 0, 0, 1)) is True
    # infeasible: grade well above threshold
    t.z[0, 2] = (MAX_GRADE_TAN + 1e-2) * h + t.z[0, 1]
    assert bool(edge_feasible(t, 0, 1, 0, 2)) is False


def test_zero_horizontal_move_infeasible():
    t = build_terrain("T1", 30)
    # degenerate move to itself has zero horizontal distance
    assert bool(edge_feasible(t, 5, 5, 5, 5)) is False


def test_path_length_3d_known_value():
    t = build_terrain("T1", 50)
    # two-cell horizontal path; length is sum of 3-D segments
    p = [(0, 0), (0, 1), (0, 2)]
    expected = 0.0
    for (i0, j0), (i1, j1) in zip(p[:-1], p[1:]):
        hd = math.hypot((j1 - j0) * t.dx, (i1 - i0) * t.dy)
        dz = abs(t.z[i1, j1] - t.z[i0, j0])
        expected += math.sqrt(hd * hd + dz * dz)
    assert math.isclose(path_length_3d(t, p), expected, rel_tol=1e-12)
    # trivial paths have zero length
    assert path_length_3d(t, None) == 0.0
    assert path_length_3d(t, [(0, 0)]) == 0.0


def _farthest_reachable(t, dist):
    n = t.n
    reachable = np.flatnonzero(np.isfinite(dist).ravel())
    goal_flat = max(reachable, key=lambda f: dist.ravel()[f])
    gi, gj = divmod(int(goal_flat), n)
    return (gi, gj)


def test_dijkstra_every_edge_in_path_feasible():
    """The reference path must obey the grade constraint on EVERY segment."""
    t = build_terrain("T4", 60)
    dist, prev = dijkstra_grade_constrained(t, t.sink_ij)
    assert np.count_nonzero(np.isfinite(dist)) > 1
    goal = _farthest_reachable(t, dist)
    path = reconstruct_path(prev, t.sink_ij, goal)
    assert path is not None and len(path) >= 2
    for (i0, j0), (i1, j1) in zip(path[:-1], path[1:]):
        assert edge_feasible(t, i0, j0, i1, j1)


def test_dijkstra_distance_matches_reconstructed_length():
    t = build_terrain("T4", 60)
    dist, prev = dijkstra_grade_constrained(t, t.sink_ij)
    goal = _farthest_reachable(t, dist)
    path = reconstruct_path(prev, t.sink_ij, goal)
    # reconstructed 3-D length must equal the Dijkstra label (optimality)
    assert math.isclose(path_length_3d(t, path), dist[goal], rel_tol=1e-9)


def test_theta_star_never_lengthens():
    """Line-of-sight smoothing can only shorten or preserve the path."""
    t = build_terrain("T4", 60)
    dist, prev = dijkstra_grade_constrained(t, t.sink_ij)
    goal = _farthest_reachable(t, dist)
    path = reconstruct_path(prev, t.sink_ij, goal)
    raw_len = path_length_3d(t, path)
    smoothed = theta_star_smooth(t, path)
    sm_len = path_length_3d(t, smoothed)
    assert sm_len <= raw_len + 1e-9
    # endpoints preserved
    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]


def test_theta_star_segments_have_line_of_sight():
    """Every smoothed segment must itself be grade-feasible (no leak)."""
    t = build_terrain("T4", 60)
    dist, prev = dijkstra_grade_constrained(t, t.sink_ij)
    goal = _farthest_reachable(t, dist)
    path = reconstruct_path(prev, t.sink_ij, goal)
    smoothed = theta_star_smooth(t, path)
    for (i0, j0), (i1, j1) in zip(smoothed[:-1], smoothed[1:]):
        assert line_of_sight(t, i0, j0, i1, j1)


def test_unreachable_goal_returns_none():
    t = build_terrain("T1", 30)
    _, prev = dijkstra_grade_constrained(t, t.sink_ij)
    # a clearly-out-of-band fake goal not in prev -> None
    assert reconstruct_path(prev, t.sink_ij, (-99, -99)) is None
