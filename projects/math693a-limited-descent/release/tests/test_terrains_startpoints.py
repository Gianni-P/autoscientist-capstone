"""Tests for terrain construction and stratified start-point selection.

Targets pitfalls: height finiteness, sink = argmin consistency, meshgrid
index convention (z[i,j] = f(xs[j], ys[i])), start-points only chosen from the
reachable set and never at the sink, and seed determinism of selection.

GridTerrain stores the RAW (scaled, un-normalised) analytic heights in ``.z``;
there is no [0,1] normalisation step. Tests therefore assert finiteness and
internal consistency rather than a unit range.
"""
import numpy as np
import pytest

from src.config import N_START_POINTS, MAX_GRADE_TAN
from src.terrains import build_terrain, terrain_function, list_terrains
from src.graph import dijkstra_grade_constrained
from src.startpoints import select_start_points


@pytest.mark.parametrize("name", list_terrains())
def test_heights_finite(name):
    t = build_terrain(name, 25)
    assert np.all(np.isfinite(t.z))
    assert t.z.shape == (25, 25)


@pytest.mark.parametrize("name", list_terrains())
def test_sink_is_global_argmin(name):
    t = build_terrain(name, 25)
    si, sj = t.sink_ij
    assert t.z[si, sj] == pytest.approx(t.z.min())


def test_meshgrid_index_convention():
    """z[i,j] must equal f(xs[j], ys[i]) per the documented convention.

    A transposed grid (x/y swapped) is a classic silent bug; assert the
    grid value matches the analytic ordering directly (no normalisation).
    """
    name = "T4"  # canonical id for the monkey-saddle-flavoured terrain
    n = 20
    t = build_terrain(name, n)
    f = terrain_function(name)
    raw = np.array([[f(t.xs[j], t.ys[i]) for j in range(n)] for i in range(n)])
    assert np.allclose(raw, t.z, atol=1e-9)


def test_unknown_terrain_raises():
    with pytest.raises(KeyError):
        terrain_function("does_not_exist")


def _reachable_mask(t):
    sink_dist, _ = dijkstra_grade_constrained(t, t.sink_ij)
    return np.isfinite(sink_dist)


def test_startpoints_are_reachable_and_not_sink():
    t = build_terrain("T1", 30)
    mask = _reachable_mask(t)
    starts = select_start_points(t, mask, seed=0)
    assert len(starts) > 0
    assert len(starts) <= N_START_POINTS
    for sp in starts:
        assert mask[sp], "selected an unreachable start point"
        assert sp != t.sink_ij, "selected the sink as a start point"


def test_startpoints_unique():
    t = build_terrain("T2", 30)
    mask = _reachable_mask(t)
    starts = select_start_points(t, mask, seed=3)
    assert len(starts) == len(set(starts))


def test_startpoint_selection_seed_determinism():
    """Same seed -> identical selection; different seed may differ."""
    t = build_terrain("T3", 30)
    mask = _reachable_mask(t)
    a = select_start_points(t, mask, seed=7)
    b = select_start_points(t, mask, seed=7)
    assert a == b, "start-point selection is not deterministic under fixed seed"


def test_empty_reachable_mask_returns_empty():
    t = build_terrain("T1", 15)
    mask = np.zeros((t.n, t.n), dtype=bool)
    assert select_start_points(t, mask, seed=1) == []
