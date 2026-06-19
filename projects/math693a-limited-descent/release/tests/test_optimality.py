"""Tests for path-length / optimality properties used as ground truth.

The optimality_gap metric compares heuristic paths against the grid reference
optimum. For that reference to be valid:
  * Theta* (any-angle) must never be LONGER than 8-connected Dijkstra when both
    are feasible -- otherwise the "optimum" is mislabeled.
  * Returned path length must equal the geometric 3-D length of the returned
    path (the reported number must match the object), guarding against a metric
    that double-counts or mismeasures edges.
  * Path endpoints must actually connect start to goal.

Start points are chosen DYNAMICALLY as a cell guaranteed reachable from the
sink (via select_reachable_start), so the tests exercise real feasible paths
instead of skipping on a permanently-infeasible fixed coordinate.
"""
import math

import numpy as np
import pytest

from src.grid_search import dijkstra, theta_star, select_reachable_start
from src.terrains import get_terrain


def _geom_length(path):
    total = 0.0
    for (x0, y0, z0), (x1, y1, z1) in zip(path[:-1], path[1:]):
        total += math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
    return total


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_theta_star_not_longer_than_dijkstra(name):
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    dij_len, dpath = dijkstra(terrain, n, start, goal)
    th_len, tpath = theta_star(terrain, n, start, goal)
    assert dpath is not None and tpath is not None
    assert math.isfinite(dij_len) and math.isfinite(th_len)
    # Any-angle path can only shorten or equal the grid path.
    assert th_len <= dij_len + 1e-6, (
        f"Theta* len {th_len} exceeds Dijkstra len {dij_len}"
    )


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_reported_length_matches_path_geometry(name):
    # Guards the path_length metric: the reported scalar must equal the actual
    # summed 3-D Euclidean length of the returned node list.
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    dij_len, dpath = dijkstra(terrain, n, start, goal)
    assert dpath is not None
    assert dij_len == pytest.approx(_geom_length(dpath), rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_path_connects_start_and_goal(name):
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    _, dpath = dijkstra(terrain, n, start, goal)
    assert dpath is not None
    # Grid snaps start/goal to nearest node; endpoints should be within one
    # grid spacing of the requested coordinates.
    spacing = 4.0 / (n - 1)
    sx, sy, _ = dpath[0]
    gx, gy, _ = dpath[-1]
    assert math.hypot(sx - start[0], sy - start[1]) <= spacing * 1.5
    assert math.hypot(gx - goal[0], gy - goal[1]) <= spacing * 1.5


def test_path_length_positive_for_distinct_endpoints():
    terrain = get_terrain("T1")
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    length, path = dijkstra(terrain, n, start, goal)
    assert path is not None
    assert length > 0.0
