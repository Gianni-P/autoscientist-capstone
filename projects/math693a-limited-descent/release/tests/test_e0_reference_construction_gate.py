"""E0 reference-construction gate: the ground-truth optimum must EXIST and the
shared setup pipeline must actually run.

This step ("E0 reference-construction review fixes") claimed three fixes:
  (1) src/common.py build_terrain(name, grid_n) -> get_terrain(name)
  (2) T5 boundary-pinned sink -> shallow off-centre quadratic bowl (interior min)
  (3) various degenerate-path E0 guards.

The project's whole objective is to compare heuristic paths against a
*constrained shortest safe path* computed as a GRID OPTIMUM (ground truth). If
that grid optimum does not exist -- because the 5-degree grade constraint
disconnects the reference graph on these analytic terrains -- then every
optimality-gap number downstream is measured against a phantom, and the
methodology is unsound regardless of how the heuristics behave.

These tests pin the failure modes I observed directly against the code:

  A. prepare_terrain must RUN. After the (1) change it calls get_terrain(name),
     which returns an analytic Terrain that has no .sink_ij / .n / .dx / .dy /
     .z -- the attributes src.graph.dijkstra_grade_constrained and
     src.startpoints.select_start_points require. So prepare_terrain raises
     AttributeError before producing a single start. A reference builder that
     cannot execute is the most basic correctness blocker.

  B. The reference graph must NOT be (near) totally disconnected. With
     MAX_GRADE_SLOPE = tan(5 deg) ~= 0.0875 and terrains whose typical gradient
     magnitude is O(1)-O(100), essentially no cell is walkable, so the sink's
     reachable set collapses to a handful of cells and NO start point gets a
     feasible reference path. We assert a non-degenerate reachable region so
     this systemic disconnection cannot pass silently.

  C. prepare_terrain must yield at least one usable start with a finite,
     positive reference (theta_len, raw_len). Zero starts means zero ground
     truth for that terrain.

  D. T5 (and every terrain's) sink must be STRICTLY interior. The claimed (2)
     fix still leaves T5's argmin at x ~= -1.99 (the domain edge), so a boundary
     artefact is being used as the optimum location.

All of these are RED against the current code on purpose: they describe the
contract the E0 reference must satisfy before any downstream experiment is
meaningful. They use small grids so the whole file runs in a few seconds.
"""
import math

import numpy as np
import pytest

from src.config import (
    TERRAINS, DOMAIN_MIN, DOMAIN_MAX, MAX_GRADE_SLOPE,
)
from src.terrains import get_terrain
from src.common import prepare_terrain
from src.grid_search import reachable_from_sink


# --------------------------------------------------------------------------- #
# A. The shared reference builder must actually execute.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", TERRAINS)
def test_prepare_terrain_runs_without_error(name):
    """prepare_terrain must not raise.

    After the build_terrain->get_terrain change it feeds an analytic Terrain
    (no .sink_ij/.n/.dx/.dy/.z) into the grid-graph Dijkstra and stratified
    start selector, which raises AttributeError. The reference builder used by
    E2..E5 must run end to end on every terrain.
    """
    setup = prepare_terrain(name, grid_n=40, seed=42)
    assert setup is not None
    assert setup.name == name
    # starts is a list (possibly empty here; emptiness is asserted separately so
    # the failure messages stay specific).
    assert isinstance(setup.starts, list)


# --------------------------------------------------------------------------- #
# B. The grade-constrained reference graph must not be (near) totally
#    disconnected around the sink.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", TERRAINS)
def test_reference_graph_not_disconnected_around_sink(name):
    """The sink's grade-feasible reachable set must be a real region.

    A 5-degree grade cap on a terrain whose |grad| is O(1)-O(100) leaves almost
    no walkable edge, so the reachable set collapses to the sink plus a couple
    of neighbours and NO start can have a feasible reference path. We require a
    meaningfully connected region (well above a single 3x3 stencil) so that a
    constrained shortest *safe* path can actually exist.
    """
    t = get_terrain(name)
    n = 120
    _, reachable = reachable_from_sink(t, n, t.sink)
    n_reach = int(reachable.sum())
    # A 3x3 stencil around the sink is at most 9 cells; anything <= that means
    # the reference graph is effectively a point and no non-trivial safe path
    # exists. Require clearly more than a single local stencil.
    assert n_reach > 25, (
        f"{name}: only {n_reach} cells reachable from the sink at n={n} under "
        f"the {MAX_GRADE_SLOPE:.4f} (tan 5deg) grade cap. The reference graph "
        "is disconnected, so the constrained shortest safe path -- the "
        "project's ground truth -- does not exist for almost any start."
    )


@pytest.mark.parametrize("name", TERRAINS)
def test_terrain_has_some_walkable_cells(name):
    """At least a non-trivial fraction of the domain must be grade-walkable.

    If virtually 0% of cells satisfy |grad| <= tan(5deg), no feasible path can
    thread the surface and the whole grade-constrained reference is vacuous.
    This is the root cause of the disconnection in (B) and is asserted
    independently so the diagnosis is unambiguous.
    """
    t = get_terrain(name)
    ax = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 160)
    gx, gy = np.meshgrid(ax, ax)
    gxv, gyv = t.grad(gx, gy)
    mag = np.hypot(np.asarray(gxv, dtype=float), np.asarray(gyv, dtype=float))
    frac = float(np.mean(mag <= MAX_GRADE_SLOPE))
    assert frac > 0.02, (
        f"{name}: only {frac*100:.3f}% of cells are grade-walkable "
        f"(|grad| <= {MAX_GRADE_SLOPE:.4f}); the surface is far too steep for a "
        "5-degree safe path to exist, so no grid optimum can be constructed."
    )


# --------------------------------------------------------------------------- #
# C. prepare_terrain must yield usable ground truth.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["T1", "T4"])
def test_prepare_terrain_yields_usable_reference(name):
    """Every kept start must carry a finite, positive (theta_len, raw_len) and
    theta_len <= raw_len. At least one start must survive.

    This is the ground-truth payload E5's optimality_gap divides by; if it is
    empty or non-finite the gap is undefined.
    """
    setup = prepare_terrain(name, grid_n=60, seed=42)
    assert len(setup.starts) > 0, (
        f"{name}: prepare_terrain produced zero usable start points -- no "
        "reference optimum exists for any start, so optimality_gap is undefined."
    )
    for sp in setup.starts:
        assert sp in setup.theta_len and sp in setup.raw_len
        rl = setup.raw_len[sp]
        tl = setup.theta_len[sp]
        assert math.isfinite(rl) and rl > 0.0
        assert math.isfinite(tl) and tl > 0.0
        # Theta* smoothing can only shorten or equal the raw Dijkstra path.
        assert tl <= rl + 1e-9, (
            f"{name}: theta_len {tl} > raw_len {rl}; smoothing must not lengthen."
        )


# --------------------------------------------------------------------------- #
# D. Sinks must be strictly interior (the claimed T5 fix).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", TERRAINS)
def test_sink_strictly_interior(name):
    """Designed sink AND the surface's grid argmin must be strictly interior.

    The (2) fix claimed to move T5's minimum off the boundary via an off-centre
    quadratic bowl, but T5's argmin is still ~(-1.99, 0.115), pinned to x=-2.
    A boundary-artefact 'global minimum' makes the optimality-gap ground truth
    a degenerate corner rather than a real basin.
    """
    t = get_terrain(name)
    margin = 0.1
    sx, sy = t.sink
    assert abs(sx) < DOMAIN_MAX - margin and abs(sy) < DOMAIN_MAX - margin, (
        f"{name}: designed sink ({sx:.3f},{sy:.3f}) is on/near the domain "
        f"boundary (|coord| must be < {DOMAIN_MAX - margin})."
    )
    n = 250
    ax = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    gx, gy = np.meshgrid(ax, ax)
    Z = np.asarray(t.height(gx, gy), dtype=float)
    idx = int(np.argmin(Z))
    mx, my = float(gx.flat[idx]), float(gy.flat[idx])
    assert abs(mx) < DOMAIN_MAX - margin and abs(my) < DOMAIN_MAX - margin, (
        f"{name}: grid global minimum ({mx:.3f},{my:.3f}) is pinned to the "
        "domain boundary; the optimum location is a boundary artefact."
    )
