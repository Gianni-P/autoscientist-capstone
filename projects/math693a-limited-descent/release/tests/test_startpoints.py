"""Tests for stratified start-point selection (src/startpoints.py).

Failure modes targeted:
  * selecting unreachable start cells (every start MUST have a feasible
    reference path or the optimality gap is undefined)
  * selecting the sink itself as a start
  * non-determinism across calls with the same seed (reproducibility)
  * seed actually influencing the selection (a frozen-RNG bug would make the
    seed argument a no-op)
"""
import numpy as np

from src.config import N_START_POINTS, START_SEED
from src.terrains import build_terrain
from src.graph import dijkstra_grade_constrained
from src.startpoints import select_start_points

SMALL_N = 70


def _reachable_mask(t):
    dist, _ = dijkstra_grade_constrained(t, t.sink_ij)
    return np.isfinite(dist)


def test_starts_are_reachable_and_not_sink():
    t = build_terrain("T1", SMALL_N)
    mask = _reachable_mask(t)
    starts = select_start_points(t, mask, seed=START_SEED)
    assert len(starts) > 0
    for (i, j) in starts:
        assert mask[i, j], "selected an unreachable start cell"
        assert (i, j) != t.sink_ij, "selected the sink as a start"


def test_starts_count_capped():
    t = build_terrain("T1", SMALL_N)
    mask = _reachable_mask(t)
    starts = select_start_points(t, mask, seed=START_SEED)
    assert len(starts) <= N_START_POINTS
    # no duplicate start cells
    assert len(set(starts)) == len(starts)


def test_selection_deterministic_same_seed():
    t = build_terrain("T1", SMALL_N)
    mask = _reachable_mask(t)
    a = select_start_points(t, mask, seed=123)
    b = select_start_points(t, mask, seed=123)
    assert a == b


def test_seed_changes_selection():
    """A different seed should generally produce a different sample.

    Guards against a bug where the RNG is created but never consumed (e.g.
    deterministic argsort), which would make the seed argument inert.
    """
    t = build_terrain("T1", SMALL_N)
    mask = _reachable_mask(t)
    a = select_start_points(t, mask, seed=1)
    b = select_start_points(t, mask, seed=999)
    # they should not be identical lists (overwhelmingly likely)
    assert a != b


def test_empty_reachable_returns_empty():
    t = build_terrain("T1", SMALL_N)
    empty = np.zeros((t.n, t.n), dtype=bool)
    assert select_start_points(t, empty, seed=START_SEED) == []
