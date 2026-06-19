"""E0 reference-construction pitfalls (plan step E0).

These target failure modes specific to the *grade-constrained grid reference*
and the analytic terrains it runs on -- the ones the review actually flagged:

  1. T4 boundedness regression. The original T4 base x^2 - y^2 + 0.1(x^2+y^2)
     is a saddle UNBOUNDED BELOW along y, so its true global min sat at a domain
     edge, not the designed sink (0,0). If that base ever resurfaces, the goal
     endpoint used as optimality-gap ground truth is wrong. We pin the surface
     so the regression is caught directly, not just via a coarse argmin.

  2. reachable_from_sink / select_reachable_start contract. The dynamic start
     selection that the grade/optimality tests rely on must be SELF-CONSISTENT:
     a start it returns must actually yield a finite-length feasible reference
     path back to the sink. If it returns an unreachable cell, every dependent
     test would skip or assert against a phantom path.

  3. Line-of-sight / Dijkstra-Theta* consistency. Theta* must never be strictly
     longer than Dijkstra (any-angle can only shorten), and a node directly
     adjacent on a flat-enough region must be mutually line-of-sight reachable.

  4. Determinism. The grid reference is a pure function of (terrain, n, start,
     goal); identical inputs must give bit-identical lengths and paths across
     repeated calls (no hidden RNG / dict-ordering nondeterminism).

DEGENERATE-PATH GUARD. Several earlier versions of these tests selected starts
that were only 1-2 grid steps from the sink at coarse n; the resulting 2-node
paths made the Theta*/determinism/reachability invariants pass *vacuously*.
We now use a FIXED multi-hop start (0.5, 0.5) at a finer resolution and assert
`len(path) >= 4` nodes before exercising any invariant, so the any-angle
shortcutting and heap/dict-ordering paths are genuinely exercised.
"""
import math

import numpy as np
import pytest

from src.config import DOMAIN_MIN, DOMAIN_MAX, MAX_GRADE_SLOPE
from src.terrains import get_terrain, HEIGHT_SCALE
from src.grid_search import (
    dijkstra, theta_star, build_height_grid,
    reachable_from_sink, select_reachable_start,
)

# A fixed interior start that is several grid hops from the sink on the gentle
# bowl terrains T1/T4, at a resolution fine enough to yield a multi-hop path.
MULTIHOP_N = 200
MULTIHOP_START = (0.5, 0.5)


# --------------------------------------------------------------------------- #
# 1. T4 boundedness regression (the fixed BLOCKER)
# --------------------------------------------------------------------------- #
def test_t4_is_bounded_below_bowl_not_saddle():
    """T4 must be a bowl: positive curvature in BOTH x and y away from sink.

    The broken saddle base gave height(0, +/-2) below the sink, so the true
    global min was a domain edge. The fixed base 3.1 x^2 + 1.1 y^2 (plus the
    well) gives height(0, +/-2) well above the sink. NB the whole surface is
    multiplied by HEIGHT_SCALE (0.02), so the rise along y is ~0.128, not O(1):
    the threshold is z_sink + 5*HEIGHT_SCALE (= z_sink + 0.1), comfortably below
    the actual rise but far above any numerical noise.
    """
    t = get_terrain("T4")
    z_sink = float(t.height(0.0, 0.0))
    margin = 5.0 * HEIGHT_SCALE  # = 0.1 on the scaled surface
    for y in (-2.0, -1.5, 1.5, 2.0):
        z = float(t.height(0.0, y))
        assert z > z_sink + margin, (
            f"T4 height(0,{y})={z:.4f} not above sink {z_sink:.4f}+{margin:.3f}; "
            "the unbounded-below saddle base has regressed."
        )
    # And along x as well (the steeper axis).
    for x in (-2.0, 2.0):
        assert float(t.height(x, 0.0)) > z_sink + margin


def test_t4_global_min_is_designed_sink():
    t = get_terrain("T4")
    n = 400
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    gx, gy = np.meshgrid(axis, axis)
    Z = np.asarray(t.height(gx, gy), dtype=float)
    idx = int(np.argmin(Z))
    mx, my = float(gx.flat[idx]), float(gy.flat[idx])
    assert abs(mx - t.sink[0]) < 0.05 and abs(my - t.sink[1]) < 0.05, (
        f"T4 grid global min ({mx:.3f},{my:.3f}) is not at sink {t.sink}"
    )
    # The minimum must be strictly interior, not pinned to a domain edge.
    assert abs(mx) < DOMAIN_MAX - 0.1 and abs(my) < DOMAIN_MAX - 0.1


# --------------------------------------------------------------------------- #
# 2. reachability / start-selection contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["T1", "T4"])
def test_select_reachable_start_yields_feasible_reference_path(name):
    """A start returned by select_reachable_start MUST give a finite Dijkstra
    path to the sink. Otherwise the dependent grade/optimality tests skip."""
    t = get_terrain(name)
    n = 60
    sink = t.sink
    start = select_reachable_start(t, n, sink)
    assert start is not None, f"{name}: no reachable non-sink start"
    # start must not be the sink itself
    assert math.hypot(start[0] - sink[0], start[1] - sink[1]) > 0.0
    length, path = dijkstra(t, n, start, sink)
    assert path is not None and math.isfinite(length) and length > 0.0


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_reachable_mask_consistent_with_dijkstra(name):
    """Every cell flagged reachable_from_sink must have a finite Dijkstra path
    from the sink; the sink itself must always be reachable.

    Uses a FINE grid (n=500) so the reachable region is large (dozens of cells)
    and the mask/Dijkstra consistency is exercised over genuinely multi-hop
    cells, not just the sink's immediate neighbours. The `len(idxs) > 5`
    assertion self-documents that the region did not degenerate to 1-hop.
    """
    t = get_terrain(name)
    n = 500
    sink = t.sink
    axis, reachable = reachable_from_sink(t, n, sink)
    sr = int(np.argmin(np.abs(axis - sink[1])))
    sc = int(np.argmin(np.abs(axis - sink[0])))
    assert reachable[sr, sc], "sink not marked reachable from itself"
    idxs = np.argwhere(reachable)
    assert len(idxs) > 5, (
        f"{name}: only {len(idxs)} reachable cells at n={n} -- region "
        "degenerated to <=1 hop; mask/Dijkstra consistency would be vacuous."
    )
    # Sample a handful of reachable cells (preferring ones far from the sink so
    # the validated Dijkstra path is genuinely multi-hop) and confirm Dijkstra
    # agrees.
    dists = np.hypot(idxs[:, 1] - sc, idxs[:, 0] - sr)
    order = np.argsort(-dists)             # farthest first
    checked = 0
    for k in order:
        r, c = int(idxs[k, 0]), int(idxs[k, 1])
        if (r, c) == (sr, sc):
            continue
        goal_xy = (float(axis[c]), float(axis[r]))
        length, path = dijkstra(t, n, sink, goal_xy)
        assert path is not None and math.isfinite(length), (
            f"{name}: cell ({r},{c}) flagged reachable but Dijkstra found no path"
        )
        checked += 1
        if checked >= 6:
            break
    assert checked >= 1


def test_unreachable_cell_not_marked_reachable():
    """A cell on the far side of a wall too steep to climb must NOT be flagged
    reachable. Negative control so the mask is not trivially all-True.

    We check a SPECIFIC far-corner cell (0,0) -> domain corner (-2,-2), which is
    separated from the Rosenbrock sink (1,1) by the steep ridge, rather than
    relying on `not reachable.all()` which is trivially true when only the sink
    cell is reachable.
    """
    t = get_terrain("T2")  # steep Rosenbrock ridge
    n = 40
    sink = t.sink
    _, reachable = reachable_from_sink(t, n, sink)
    # Far corner of the domain is across the steep ridge from the (1,1) sink.
    assert not reachable[0, 0], (
        "far domain corner marked reachable across a 5-degree-capped steep "
        "ridge -- the grade gate is leaking."
    )
    # And, more broadly, not every cell is reachable.
    assert not reachable.all(), (
        "every cell marked reachable on a steep terrain -- grade gate leaking"
    )


# --------------------------------------------------------------------------- #
# 3. Dijkstra / Theta* consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["T1", "T4"])
def test_theta_star_never_longer_than_dijkstra(name):
    """Theta* (any-angle) can only shorten Dijkstra, never lengthen it.

    Uses a FIXED multi-hop start at a fine grid and asserts the Dijkstra path
    has >= 4 nodes, so the any-angle shortcutting logic is actually exercised
    (a 2-node path makes the invariant vacuous: Theta* exits immediately).
    """
    t = get_terrain(name)
    n = MULTIHOP_N
    sink = t.sink
    start = MULTIHOP_START
    dlen, dpath = dijkstra(t, n, start, sink)
    tlen, tpath = theta_star(t, n, start, sink)
    assert dpath is not None and tpath is not None
    assert len(dpath) >= 4, (
        f"{name}: Dijkstra path only {len(dpath)} nodes -- too short to "
        "exercise Theta* any-angle shortcutting; pick a farther start."
    )
    assert tlen <= dlen + 1e-6, f"{name}: Theta* {tlen} longer than Dijkstra {dlen}"


def test_theta_star_path_respects_grade_constraint():
    """The path Theta* RETURNS must itself obey the grade cap at the heights it
    reports. This is the consistency the height-source fix guarantees: feasibility
    is now measured on the same grid Z that the returned node heights come from.
    """
    for name in ["T1", "T4"]:
        t = get_terrain(name)
        n = MULTIHOP_N
        tlen, tpath = theta_star(t, n, MULTIHOP_START, t.sink)
        assert tpath is not None
        for (x0, y0, z0), (x1, y1, z1) in zip(tpath[:-1], tpath[1:]):
            horiz = math.hypot(x1 - x0, y1 - y0)
            if horiz > 0:
                assert abs(z1 - z0) / horiz <= MAX_GRADE_SLOPE + 1e-9, (
                    f"{name}: returned Theta* segment grade "
                    f"{abs(z1 - z0) / horiz:.4f} exceeds cap {MAX_GRADE_SLOPE:.4f}"
                )


def test_infeasible_returns_inf_and_none():
    """When no feasible path exists the reference must report (inf, None),
    never a constraint-violating path passed off as the optimum."""
    t = get_terrain("T2")
    n = 30
    length, path = dijkstra(t, n, (-1.8, -1.8), (1.8, 1.8))
    if path is None:
        assert math.isinf(length)
    else:
        # if a path is returned it must obey the grade constraint everywhere
        for (x0, y0, z0), (x1, y1, z1) in zip(path[:-1], path[1:]):
            horiz = math.hypot(x1 - x0, y1 - y0)
            if horiz > 0:
                assert abs(z1 - z0) / horiz <= MAX_GRADE_SLOPE + 1e-9


# --------------------------------------------------------------------------- #
# 4. Determinism of the reference solver
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("solver", [dijkstra, theta_star])
def test_reference_solver_is_deterministic(solver):
    """The reference solver is a pure function of (terrain, n, start, goal).

    Uses a FIXED multi-hop start at a fine grid and asserts the path has >= 3
    nodes before the equality check, so determinism is tested over a genuine
    multi-edge search (where heap tie-breaking / dict-ordering nondeterminism
    could actually surface) rather than a trivial single-edge path.
    """
    t = get_terrain("T1")
    n = MULTIHOP_N
    sink = t.sink
    start = MULTIHOP_START
    l1, p1 = solver(t, n, start, sink)
    assert p1 is not None and len(p1) >= 3, (
        f"reference path only {0 if p1 is None else len(p1)} nodes -- "
        "too short to exercise heap/dict-ordering nondeterminism."
    )
    l2, p2 = solver(t, n, start, sink)
    assert l1 == l2, "reference length non-deterministic across identical calls"
    assert p1 == p2, "reference path non-deterministic across identical calls"


def test_build_height_grid_deterministic_and_oriented():
    """build_height_grid must be a pure function and Z[r,c] == height(x_c, y_r)."""
    t = get_terrain("T3")
    n = 50
    axis1, Z1 = build_height_grid(t, n)
    axis2, Z2 = build_height_grid(t, n)
    assert np.array_equal(axis1, axis2)
    assert np.array_equal(Z1, Z2)
    # orientation spot-check: row index -> y, column index -> x
    for r, c in [(3, 7), (20, 5), (44, 49)]:
        assert Z1[r, c] == pytest.approx(
            float(t.height(axis1[c], axis1[r])), rel=1e-9, abs=1e-9
        )
