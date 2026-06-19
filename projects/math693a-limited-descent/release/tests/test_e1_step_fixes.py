"""Tests for the E1 fixes / terrain-shape guarantees in this plan step.

Plan items under test:
  (1) gate the T1 (elliptic_paraboloid) geodesic-deviation internal-validity
      check on direct-start->sink grade feasibility, via the
      ``direct_path_grade()`` helper in experiment_e1;
  (2) T4 (monkey-saddle-flavoured) must form a *genuinely 2-D* grade-feasible
      reachable basin around its interior sink -- not a degenerate one-row
      sliver;
  (3) T3 (sinusoidal_valley) must carry a visible sinusoidal ripple, so the
      surface is genuinely multi-basin (not a smooth bowl indistinguishable
      from T1).

Terrains are keyed by their canonical T-codes (T1..T5). These complement the
generic pitfall suite, which already covers reachability, seed liveness, and
reference-path grade integrity.
"""
import json
import math
import os

import numpy as np
import pytest

from src.config import MAX_GRADE_TAN, DOMAIN_MIN, DOMAIN_MAX
from src.terrains import build_terrain, terrain_function
from src.graph import dijkstra_grade_constrained
from src.experiment_e1 import direct_path_grade, run_e1


GRID_N_SMALL = 40


def _reachable_mask(t):
    sink_dist, _ = dijkstra_grade_constrained(t, t.sink_ij)
    return np.isfinite(sink_dist)


# ---------------------------------------------------------------------------
# Fix (1): direct_path_grade helper + T1 gating
# ---------------------------------------------------------------------------
def test_direct_path_grade_matches_definition():
    """direct_path_grade == |dz| / horizontal Euclidean distance start->sink.

    Recompute the quantity independently from terrain fields and require an
    exact match, so a refactor that swaps numerator/denominator or uses the
    wrong distance is caught.
    """
    t = build_terrain("T2", GRID_N_SMALL)
    gi, gj = t.sink_ij
    for start in [(0, 0), (0, GRID_N_SMALL - 1), (GRID_N_SMALL - 1, 0),
                  (GRID_N_SMALL // 2, GRID_N_SMALL // 3)]:
        si, sj = start
        dxw = (gj - sj) * t.dx
        dyw = (gi - si) * t.dy
        h = math.hypot(dxw, dyw)
        dz = abs(t.z[gi, gj] - t.z[si, sj])
        expected = dz / h if h > 0 else 0.0
        assert direct_path_grade(t, start) == pytest.approx(expected, rel=1e-12)


def test_direct_path_grade_zero_at_sink():
    """No horizontal separation (start == sink) must give grade 0.0, not NaN/inf.

    Guards the degenerate-divide branch that the gating logic depends on.
    """
    t = build_terrain("T1", GRID_N_SMALL)
    g = direct_path_grade(t, t.sink_ij)
    assert g == 0.0
    assert math.isfinite(g)


def test_direct_path_grade_nonnegative_and_finite():
    """Grade is a magnitude ratio: always finite and >= 0 for every cell."""
    t = build_terrain("T4", GRID_N_SMALL)
    for i in range(0, GRID_N_SMALL, 7):
        for j in range(0, GRID_N_SMALL, 7):
            g = direct_path_grade(t, (i, j))
            assert math.isfinite(g)
            assert g >= 0.0


def test_t1_geodesic_check_gated_on_direct_feasibility(tmp_path, monkeypatch):
    """The T1 geodesic-deviation failure must only fire for direct-feasible starts.

    This is the heart of fix (1): a start whose straight start->sink line is
    too steep (direct_path_grade > tan(5 deg)) is *required* to detour, so its
    geodesic deviation is physically meaningless and must NOT be recorded as a
    t1_geodesic internal-validity failure. We assert that property directly on
    the run output instead of trusting the flag in isolation.
    """
    monkeypatch.chdir(tmp_path)
    summary = run_e1(run_id="gate", seed=0, grid_n=GRID_N_SMALL)
    t1_failures = [
        f for f in summary["validity_failures"]
        if f.get("check") == "t1_geodesic"
    ]
    t = build_terrain("T1", GRID_N_SMALL)
    for f in t1_failures:
        si, sj = f["start"]
        assert direct_path_grade(t, (si, sj)) <= MAX_GRADE_TAN + 1e-12, (
            "a t1_geodesic failure was logged for a start whose direct "
            "line is grade-INFEASIBLE -- gating is broken"
        )


def test_t1_records_carry_direct_path_fields(tmp_path, monkeypatch):
    """Each E1 trial must record direct_path_grade and direct_path_feasible,
    and the boolean must be consistent with the grade vs the threshold."""
    monkeypatch.chdir(tmp_path)
    run_e1(run_id="fields", seed=0, grid_n=GRID_N_SMALL)
    with open(os.path.join(tmp_path, "runs", "fields", "e1_trials.jsonl")) as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    assert recs, "no trial records produced"
    for r in recs:
        assert "direct_path_grade" in r
        assert "direct_path_feasible" in r
        assert isinstance(r["direct_path_feasible"], bool)
        expected = r["direct_path_grade"] <= MAX_GRADE_TAN
        assert r["direct_path_feasible"] == expected


# ---------------------------------------------------------------------------
# Fix (3): T3 (sinusoidal_valley) carries a real sinusoidal ripple
# ---------------------------------------------------------------------------
def test_t3_sinusoidal_ripple_present():
    """The sin(3x)*cos(3y) ripple must dominate the gentle quadratic term.

    Compare the raw analytic surface (before the global HEIGHT_SCALE) against
    the pure quadratic part 0.1*(x**2+y**2) at a point where
    sin(3x)*cos(3y) ~ +1. The deviation must be ~1 (the ripple amplitude), not
    ~0; a vanishing ripple would make T3 a smooth bowl.
    """
    f = terrain_function("T3")  # returns SCALED height f+g (incl. well)
    # Evaluate away from the Gaussian well so the ripple is the dominant signal.
    # Pick x with sin(3x) ~ 1 and y with cos(3y) ~ 1, far from the sink.
    x, y = math.pi / 6.0, 0.0  # sin(pi/2)=1, cos(0)=1
    # The ripple's contribution to the *raw* surface is ~ +1.0 at this point;
    # after HEIGHT_SCALE (0.02) it is ~0.02. We require the actual surface to
    # differ from the pure quadratic-only surface by at least half the ripple.
    quad_only_raw = 0.1 * (x ** 2 + y ** 2)
    # scaled pure-quadratic value (no ripple, no well)
    from src.terrains import HEIGHT_SCALE
    quad_only = HEIGHT_SCALE * quad_only_raw
    deviation = abs(float(f(x, y)) - quad_only)
    assert deviation > 0.5 * HEIGHT_SCALE, (
        f"T3 ripple deviation {deviation:.5f} too small -- the sinusoidal "
        "ripple appears to be absent (surface is a smooth bowl)"
    )


def test_t3_distinct_from_smooth_bowl():
    """The rippled surface must differ from a pure quadratic bowl across the grid.

    A near-zero ripple would make T3 indistinguishable from a single-basin bowl;
    require a visible max deviation on the grid.
    """
    from src.terrains import HEIGHT_SCALE
    f = terrain_function("T3")
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, 60)
    X, Y = np.meshgrid(xs, xs)
    surface = np.asarray(f(X, Y), dtype=float)
    quad = HEIGHT_SCALE * 0.1 * (X ** 2 + Y ** 2)
    max_dev = float(np.max(np.abs(surface - quad)))
    assert max_dev > 0.01, (
        f"max ripple deviation {max_dev:.5f} too small -- T3 is "
        "indistinguishable from a smooth bowl"
    )


# ---------------------------------------------------------------------------
# Fix (2): T4 must form a genuinely 2-D reachable basin
# ---------------------------------------------------------------------------
def test_t4_basin_is_two_dimensional():
    """The T4 reachable region must span more than one grid row AND column.

    A reachable set confined to a single grid row (or column) means the
    interior-minimum basin is a degenerate 1-D sliver rather than a genuine 2-D
    basin around the sink.
    """
    t = build_terrain("T4", 150)
    mask = _reachable_mask(t)
    ii, jj = np.where(mask)
    assert ii.size > 0, "T4 has no reachable cells at all"
    n_rows = len(set(ii.tolist()))
    n_cols = len(set(jj.tolist()))
    assert n_rows > 1 and n_cols > 1, (
        f"T4 reachable region spans {n_rows} row(s) x {n_cols} col(s) -- "
        "still a degenerate 1-D sliver, not the intended 2-D basin"
    )


def test_t4_reachable_cells_exceed_target():
    """T4 must have substantially more than one reachable cell at a fine grid.

    A sliver of a handful of reachable cells (a degenerate basin) must fail;
    we require a comfortably 2-D basin (> 100 cells) at a 200x200 grid, which
    is a single Dijkstra well within the time budget.
    """
    t = build_terrain("T4", 200)
    mask = _reachable_mask(t)
    reachable = int(mask.sum())
    assert reachable > 100, (
        f"T4 has only {reachable} reachable cells at the 200x200 grid; the "
        "interior-minimum basin is too small to be a genuine 2-D basin"
    )
