"""Regression tests for the revision fixes (handoff targets).

These assert the THREE behavioural fixes the revision made, each of which a
prior version got wrong and the reviewer flagged:

  1. T2 walkability (HEIGHT_SCALE_T2 = 0.0005). The Rosenbrock ridge has
     O(100)-O(500) gradients; under the global HEIGHT_SCALE (0.02) only the
     sink cell is reachable (the constrained reference path does not exist).
     With the dedicated T2 scale a genuinely 2-D walkable region exists and
     grows with grid resolution -- so reachable_cells >> 1 and is NOT just the
     single sink cell.

  2. T3 unique-minimum depth margin. The sinusoidal valley has 10+ competing
     basins; the well amplitude (-25.0) must make the designed sink beat the
     second-lowest basin by more than UNIQUE_MIN_DEPTH_MARGIN (0.3) AFTER the
     global HEIGHT_SCALE, otherwise verify_unique_minimum fails and the ground
     truth is ambiguous.

  3. E5 NONTRIVIAL_TERRAINS canonical T-codes. The pooled-nontrivial H1 test
     loops `if name in NONTRIVIAL_TERRAINS` over RUN_TERRAINS (T1..T5). If
     NONTRIVIAL_TERRAINS held legacy names ('rosenbrock_ridge', ...) the loop
     matches nothing and the pooled hypothesis is structurally untestable
     (pooled CI is None / n == 0). The fix uses canonical ['T2','T3','T4'].
"""
import numpy as np
import pytest

from src.config import UNIQUE_MIN_DEPTH_MARGIN
from src.terrains import (
    build_terrain, get_terrain, Terrain, GridTerrain, _gauss, _x, _y,
    HEIGHT_SCALE, HEIGHT_SCALE_T2,
)
from src.graph import dijkstra_grade_constrained
from src.validation import verify_unique_minimum
from src.experiment_e5 import NONTRIVIAL_TERRAINS
from src.common import RUN_TERRAINS


def _reachable_count(terrain):
    dist, _ = dijkstra_grade_constrained(terrain, terrain.sink_ij)
    return int(np.isfinite(dist).sum())


# ---------------------------------------------------------------------------
# FIX 1: T2 walkability under the dedicated HEIGHT_SCALE_T2.
# ---------------------------------------------------------------------------
def test_t2_dedicated_scale_is_much_smaller_than_global():
    # The whole reason T2 needs its own scale: its raw gradients are ~20-40x
    # larger than the rest, so HEIGHT_SCALE_T2 must be well below HEIGHT_SCALE.
    assert HEIGHT_SCALE_T2 < HEIGHT_SCALE
    assert HEIGHT_SCALE_T2 == pytest.approx(0.0005)


def test_t2_has_nontrivial_walkable_region():
    # With the fix, T2 has a genuine 2-D walkable region: far more than the
    # single sink cell is reachable on the grade-constrained graph.
    t = build_terrain("T2", 120)
    reach = _reachable_count(t)
    assert reach > 50, (
        f"T2 only {reach} cells reachable -- HEIGHT_SCALE_T2 fix appears lost; "
        "the constrained reference path collapses to the sink cell."
    )


def test_t2_walkable_region_grows_with_resolution():
    # A genuine 2-D walkable patch contains more cells as the grid refines.
    # A spurious "1 reachable cell" artefact would stay pinned near 1.
    r_coarse = _reachable_count(build_terrain("T2", 80))
    r_fine = _reachable_count(build_terrain("T2", 160))
    assert r_coarse > 1
    assert r_fine > r_coarse, (
        f"T2 reachable cells did not grow with resolution "
        f"({r_coarse} -> {r_fine}); region is not a real 2-D patch."
    )


def test_t2_global_scale_would_leave_only_sink_reachable():
    # Direct contrast proving the fix matters: if T2 were built with the
    # GLOBAL HEIGHT_SCALE (the pre-fix bug) only the sink cell is reachable.
    f2 = (1 - _x) ** 2 + 100 * (_y - _x ** 2) ** 2
    g2 = _gauss(1.0, 1.0, -300.0, 0.4)
    bad = Terrain("T2_globalscale", HEIGHT_SCALE * (f2 + g2), (1.0, 1.0))
    bad_grid = GridTerrain(bad, 120)
    assert _reachable_count(bad_grid) <= 2, (
        "Sanity check failed: global-scale T2 should be essentially "
        "disconnected (only the sink reachable)."
    )
    # And the real (fixed) T2 must be dramatically more connected.
    assert _reachable_count(build_terrain("T2", 120)) > 20 * 2


# ---------------------------------------------------------------------------
# FIX 2: T3 unique-minimum depth margin.
# ---------------------------------------------------------------------------
def test_t3_unique_minimum_margin_exceeds_threshold():
    # Use a moderate grid (the production check uses GRID_UNIQUEMIN=2000 but
    # the T3 margin is already well-separated at 400, since the well dominates
    # the sinusoidal basins by ~0.5 after the global HEIGHT_SCALE).
    res = verify_unique_minimum(get_terrain("T3"), n=400)
    assert res["passed"], f"T3 unique-min check failed: {res}"
    assert res["depth_margin"] > UNIQUE_MIN_DEPTH_MARGIN, (
        f"T3 depth_margin {res['depth_margin']:.3f} <= required "
        f"{UNIQUE_MIN_DEPTH_MARGIN}; the well no longer dominates the "
        "second-lowest sinusoidal basin."
    )
    # The plan margin is ~0.5 with amp -25.0; require comfortably above 0.3.
    assert res["depth_margin"] > 0.3


def test_t3_well_amplitude_drives_the_margin():
    # Guard the specific fix: a far shallower well (the pre-fix amplitude would
    # have been too small) must FAIL the margin, while the shipped T3 passes.
    # Reconstruct T3's base + a deliberately shallow well and confirm it is the
    # well depth -- not luck -- that produces the passing margin.
    import sympy as sp
    from src.terrains import _coarse_argmin
    f3 = sp.sin(3 * _x) * sp.cos(3 * _y) + 0.1 * (_x ** 2 + _y ** 2)
    f3_fn = sp.lambdify((_x, _y), f3, modules="numpy")
    cx, cy = _coarse_argmin(f3_fn)
    shallow = _gauss(cx, cy, -1.0, 0.3)   # far too shallow well
    weak = Terrain("T3_shallow", HEIGHT_SCALE * (f3 + shallow), (cx, cy))
    res_weak = verify_unique_minimum(weak, n=400)
    assert not res_weak["passed"], (
        "A deliberately shallow T3 well should FAIL the unique-min margin; "
        "if it passes the check is not actually discriminating."
    )
    assert verify_unique_minimum(get_terrain("T3"), n=400)["passed"]


# ---------------------------------------------------------------------------
# FIX 3: E5 NONTRIVIAL_TERRAINS uses canonical T-codes that the pooled loop
#        can actually match against RUN_TERRAINS.
# ---------------------------------------------------------------------------
def test_nontrivial_terrains_are_canonical_tcodes():
    assert NONTRIVIAL_TERRAINS == ["T2", "T3", "T4"], (
        f"NONTRIVIAL_TERRAINS={NONTRIVIAL_TERRAINS} are not the canonical "
        "T-codes; the pre-fix legacy names ('rosenbrock_ridge', ...) make the "
        "pooled H1 loop match nothing."
    )


def test_nontrivial_terrains_subset_of_run_terrains():
    # The pooled H1 loop iterates RUN_TERRAINS and tests membership in
    # NONTRIVIAL_TERRAINS. If any NONTRIVIAL id is absent from RUN_TERRAINS the
    # corresponding terrain's COG values are silently never pooled.
    assert set(NONTRIVIAL_TERRAINS).issubset(set(RUN_TERRAINS)), (
        f"NONTRIVIAL_TERRAINS {NONTRIVIAL_TERRAINS} not a subset of "
        f"RUN_TERRAINS {RUN_TERRAINS}; pooled-nontrivial H1 would be empty."
    )
    # Must be non-empty -- an empty pool makes the core hypothesis untestable.
    assert len(NONTRIVIAL_TERRAINS) >= 1


def test_pooled_loop_matches_every_nontrivial_terrain():
    # Mirror the exact membership test run_e5 uses for pooling and assert each
    # nontrivial terrain is actually reached by it. This catches a name-format
    # mismatch (e.g. casing / legacy alias) even if both lists are non-empty.
    matched = [name for name in RUN_TERRAINS if name in NONTRIVIAL_TERRAINS]
    assert sorted(matched) == sorted(NONTRIVIAL_TERRAINS), (
        f"pooled loop matched {matched} but NONTRIVIAL_TERRAINS is "
        f"{NONTRIVIAL_TERRAINS}; the H1 pool would drop terrains."
    )
