"""Tests for analytic terrain construction (src/terrains.py).

NOTE: this file previously targeted a removed terrain API (build_terrain(name, N)
returning a normalised .z grid with .sink_ij, plus terrain names like
'curved_valley'). The terrain module was rewritten to expose analytic SymPy
terrains (Terrain.height / Terrain.grad / Terrain.sink) over the canonical
T1..T5 suite, plus a GridTerrain returned by build_terrain(name, grid_n).

Failure modes targeted (all named in the E0 plan):
  * the canonical accessors must return the five-terrain T1..T5 suite
  * height() must broadcast elementwise consistently with scalar evaluation
    (a meshgrid/transpose bug silently corrupts every gradient and path)
  * grad() must return the analytic partials (orientation: dz/dx, dz/dy)
  * each Terrain.sink must coincide with the surface's grid argmin AND be
    strictly interior to the domain (a sink pinned to the boundary makes the
    optimality-gap ground truth degenerate)
"""
import math

import numpy as np
import pytest

from src.config import TERRAINS, DOMAIN_MIN, DOMAIN_MAX
from src.terrains import (
    get_terrain, all_terrains, build_terrain, terrain_function, list_terrains,
    GridTerrain,
)

SMALL_N = 80


def test_list_terrains_matches_config():
    assert list_terrains() == list(TERRAINS)
    for name in TERRAINS:
        # terrain_function must return the terrain's own callable height: same
        # behaviour as Terrain.height (compare values, not method identity --
        # bound methods are fresh objects each access).
        fn = terrain_function(name)
        assert callable(fn)
        t = get_terrain(name)
        for x, y in [(0.1, -0.2), (-0.5, 0.5)]:
            assert float(fn(x, y)) == pytest.approx(float(t.height(x, y)))


def test_all_terrains_canonical_order_and_names():
    ts = all_terrains()
    assert [t.name for t in ts] == ["T1", "T2", "T3", "T4", "T5"]


def test_build_terrain_returns_grid_terrain():
    # build_terrain(name, grid_n) is the canonical grid entry point: it returns
    # a GridTerrain (NOT the analytic Terrain returned by get_terrain), sampled
    # on grid_n^2. It is a *distinct* object from the analytic terrain.
    gt = build_terrain("T1", SMALL_N)
    assert isinstance(gt, GridTerrain)
    assert gt is not get_terrain("T1")
    assert gt.n == SMALL_N
    assert gt.z.shape == (SMALL_N, SMALL_N)
    # the GridTerrain wraps the same analytic terrain identity (same name).
    assert gt.name == get_terrain("T1").name


@pytest.mark.parametrize("name", TERRAINS)
def test_height_broadcasts_consistently_with_scalar(name):
    """height() on a meshgrid must equal scalar height() at each (x,y).

    A meshgrid X/Y transpose leaves the value set intact but flips the surface,
    silently corrupting every gradient and path. We build Z over a meshgrid and
    compare against an explicit double loop using axis[col]=x, axis[row]=y.
    """
    t = get_terrain(name)
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, SMALL_N)
    gx, gy = np.meshgrid(axis, axis)        # gx varies along columns, gy rows
    Z = np.asarray(t.height(gx, gy), dtype=float)
    assert Z.shape == (SMALL_N, SMALL_N)
    # Spot-check several cells with explicit scalar evaluation.
    rng = np.random.default_rng(0)
    for _ in range(40):
        i = int(rng.integers(SMALL_N))
        j = int(rng.integers(SMALL_N))
        scalar = float(t.height(axis[j], axis[i]))  # x=axis[col], y=axis[row]
        assert Z[i, j] == pytest.approx(scalar, rel=1e-9, abs=1e-9), (
            f"{name}: height meshgrid[{i},{j}] disagrees with scalar eval "
            "-- likely an X/Y meshgrid transpose."
        )


@pytest.mark.parametrize("name", TERRAINS)
def test_grad_orientation_matches_finite_difference_sign(name):
    """grad returns (dz/dx, dz/dy) in that order, not swapped.

    A swapped-axis gradient passes an isotropic FD-magnitude check but points
    the wrong way on anisotropic terrains. We compare each component against the
    directional finite difference along its own axis at a few points.
    """
    t = get_terrain(name)
    eps = 1e-5
    for x, y in [(0.3, -0.4), (-0.7, 0.6), (0.9, 0.2)]:
        gx, gy = t.grad(x, y)
        fd_x = (float(t.height(x + eps, y)) - float(t.height(x - eps, y))) / (2 * eps)
        fd_y = (float(t.height(x, y + eps)) - float(t.height(x, y - eps))) / (2 * eps)
        assert float(gx) == pytest.approx(fd_x, abs=1e-3), f"{name}: dz/dx mismatch"
        assert float(gy) == pytest.approx(fd_y, abs=1e-3), f"{name}: dz/dy mismatch"


@pytest.mark.parametrize("name", TERRAINS)
def test_sink_is_grid_argmin(name):
    """Terrain.sink must point at (close to) the surface's grid global minimum
    AND that minimum must be strictly interior to the domain.

    A sink pinned to a domain edge (the T5 linear-tilt bug) means the designed
    "global minimum" is an artefact of the boundary, not a real basin, which
    makes the optimality-gap ground truth degenerate.
    """
    t = get_terrain(name)
    n = 300
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    gx, gy = np.meshgrid(axis, axis)
    Z = np.asarray(t.height(gx, gy), dtype=float)
    idx = int(np.argmin(Z))
    mx, my = float(gx.flat[idx]), float(gy.flat[idx])
    sx, sy = t.sink
    # within a couple grid cells of the designed sink
    tol = 3.0 * (DOMAIN_MAX - DOMAIN_MIN) / (n - 1) + 0.05
    assert math.hypot(mx - sx, my - sy) <= tol, (
        f"{name}: grid argmin ({mx:.3f},{my:.3f}) far from designed sink "
        f"({sx:.3f},{sy:.3f})"
    )
    # Interior-sink guard for ALL terrains (catches T5 boundary-pinned min).
    assert abs(sx) < DOMAIN_MAX - 0.1 and abs(sy) < DOMAIN_MAX - 0.1, (
        f"{name}: sink ({sx:.3f},{sy:.3f}) is on/near the domain boundary "
        f"|x|<{DOMAIN_MAX - 0.1}; designed minimum must be strictly interior."
    )
    assert abs(mx) < DOMAIN_MAX - 0.1 and abs(my) < DOMAIN_MAX - 0.1, (
        f"{name}: grid argmin ({mx:.3f},{my:.3f}) is on/near the domain "
        "boundary; global minimum must be strictly interior."
    )


def test_unknown_terrain_raises():
    with pytest.raises(KeyError):
        get_terrain("no_such_terrain")
    with pytest.raises(KeyError):
        terrain_function("no_such_terrain")
