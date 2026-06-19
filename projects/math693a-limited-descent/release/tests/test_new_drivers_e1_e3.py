"""Tests for the freshly-revised E1 / E3 drivers and the terrain-keying fixes.

These target the specific pitfalls in this revision (per the code_gen handoff
and the project watch_items):

  * E3 SENSITIVITY_TERRAIN and E1 ELLIPTIC_PARABOLOID must be CANONICAL terrain
    keys ('T2' / 'T1'); the prior values ('rosenbrock_ridge', 'elliptic_paraboloid')
    were not keys returned by list_terrains(), so prepare_terrain raised KeyError
    (E3) and the T1 geodesic check was a permanently-dead branch (E1).
  * prepare_terrain(SENSITIVITY_TERRAIN, ...) must not raise -- a stale key would
    KeyError at run time, aborting the whole sensitivity sub-experiment.
  * The E3 sensitivity sub-experiment is keyed to ONE terrain; if that terrain
    yields zero reachable start points at the grid used, the sub-experiment is
    silently empty (no trials) and the step-size science vanishes. This is the
    watch_item: assert the chosen sensitivity terrain actually produces starts.
  * E1's geodesic-deviation check (b) must be ACTIVE on T1 (the branch must be
    reachable: ELLIPTIC_PARABOLOID must be a real terrain name E1 iterates over).

We use a small grid so the suite stays fast.
"""
import math
import os

import pytest

from src.terrains import list_terrains
from src.common import prepare_terrain
from src.experiment_e1 import (
    run_e1, direct_path_grade, ELLIPTIC_PARABOLOID,
)
from src.experiment_e3 import (
    SENSITIVITY_TERRAIN, SEEDS, SENSITIVITY_DS, ROTATION_VARIANTS,
)
from src.strategies import STRATEGIES
from src.config import MAX_GRADE_TAN

SMALL_GRID = 40


# --------------------------------------------------------------------------
# Terrain-keying fixes: the canonical-id constants MUST be real terrain keys.
# --------------------------------------------------------------------------
def test_sensitivity_terrain_is_a_canonical_key():
    """E3's sensitivity terrain must be one of the keys list_terrains() exposes;
    a stale alias ('rosenbrock_ridge') makes prepare_terrain raise KeyError."""
    assert SENSITIVITY_TERRAIN in list_terrains(), (
        f"{SENSITIVITY_TERRAIN!r} is not a canonical terrain key "
        f"{list_terrains()}")


def test_elliptic_paraboloid_is_a_canonical_key():
    """E1's T1 geodesic check is gated on name == ELLIPTIC_PARABOLOID; if that
    constant is not a name E1 iterates over the check (b) is dead code."""
    assert ELLIPTIC_PARABOLOID in list_terrains(), (
        f"{ELLIPTIC_PARABOLOID!r} is not a canonical terrain key "
        f"{list_terrains()}")


def test_prepare_terrain_on_sensitivity_terrain_does_not_raise():
    """The exact call the E3 sensitivity loop makes must succeed -- a stale key
    would KeyError here and abort the whole sub-experiment at run time."""
    setup = prepare_terrain(SENSITIVITY_TERRAIN, SMALL_GRID, 0)
    assert setup.name == SENSITIVITY_TERRAIN
    # terrain grid API the downstream consumers rely on
    assert setup.terrain.n == SMALL_GRID
    assert hasattr(setup.terrain, "sink_ij")


# --------------------------------------------------------------------------
# Watch_item: the sensitivity sub-experiment must actually have trials.
# --------------------------------------------------------------------------
def test_sensitivity_terrain_yields_start_points():
    """The step-size sensitivity sub-experiment loops over setup.starts on the
    sensitivity terrain; if that terrain has zero reachable starts at the grid
    used, the sub-experiment is silently empty and produces no science.

    This is the documented watch_item (HEIGHT_SCALE=0.02 was validated only
    analytically; a terrain with 0 reachable starts gives an empty reference).
    """
    found = False
    for sd in SEEDS:
        setup = prepare_terrain(SENSITIVITY_TERRAIN, SMALL_GRID, sd)
        if len(setup.starts) > 0:
            found = True
            break
    assert found, (
        f"sensitivity terrain {SENSITIVITY_TERRAIN!r} produced ZERO reachable "
        f"start points at grid_n={SMALL_GRID} across seeds {SEEDS}: the E3 "
        f"step-size sub-experiment would record no trials and the optimality "
        f"gap reference is empty for it")


# --------------------------------------------------------------------------
# E3 config sanity (the revision must not have re-broken these).
# --------------------------------------------------------------------------
def test_rotation_variants_are_exactly_the_two_directions():
    assert set(ROTATION_VARIANTS) == {"rotation_cw", "rotation_ccw"}
    for v in ROTATION_VARIANTS:
        assert v in STRATEGIES


def test_sensitivity_ds_distinct_and_positive():
    assert len(set(SENSITIVITY_DS)) == len(SENSITIVITY_DS)
    assert all(d > 0 for d in SENSITIVITY_DS)


# --------------------------------------------------------------------------
# direct_path_grade contract (gates E1 check (b)).
# --------------------------------------------------------------------------
def test_direct_path_grade_zero_at_sink():
    """Start == sink has no horizontal separation -> grade defined as 0.0
    (must not divide by zero)."""
    setup = prepare_terrain("T1", SMALL_GRID, 0)
    t = setup.terrain
    assert direct_path_grade(t, t.sink_ij) == 0.0


def test_direct_path_grade_matches_manual_slope():
    """Grade must equal |dz| / horizontal for a known off-sink cell."""
    setup = prepare_terrain("T1", SMALL_GRID, 0)
    t = setup.terrain
    si, sj = t.sink_ij
    # pick a cell offset diagonally but inside the grid
    pi = min(si + 3, t.n - 1)
    pj = min(sj + 3, t.n - 1)
    if (pi, pj) == (si, sj):
        pytest.skip("degenerate offset")
    dxw = (sj - pj) * t.dx
    dyw = (si - pi) * t.dy
    horiz = math.hypot(dxw, dyw)
    dz = abs(t.z[si, sj] - t.z[pi, pj])
    expected = dz / horiz
    assert direct_path_grade(t, (pi, pj)) == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------
# E1 end-to-end: T1 check (b) must be active (not dead) and validity must hold.
# --------------------------------------------------------------------------
def test_run_e1_t1_geodesic_check_is_reachable(tmp_path):
    """With ELLIPTIC_PARABOLOID correctly keyed to a real terrain, E1 actually
    visits T1 and computes the geodesic deviation; the run must complete and
    report T1 in its terrain summaries with >0 starts (so the check (b) branch
    is genuinely exercised, not skipped because T1 was never built)."""
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        summary = run_e1(run_id="t1geo", seed=0, grid_n=SMALL_GRID)
    finally:
        os.chdir(cwd)
    t1_summaries = [s for s in summary["terrain_summaries"]
                    if s["terrain"] == ELLIPTIC_PARABOLOID]
    assert t1_summaries, "E1 never produced a summary for T1"
    assert t1_summaries[0]["n_starts"] > 0, (
        "T1 had no starts -> geodesic check (b) could never fire")


def test_run_e1_reports_direct_path_feasibility_field(tmp_path):
    """Every E1 trial must carry direct_path_grade / direct_path_feasible so the
    geodesic check can be gated; feasibility must agree with the threshold."""
    import json
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_e1(run_id="dpf", seed=0, grid_n=SMALL_GRID)
        path = os.path.join(tmp_path, "runs", "dpf", "e1_trials.jsonl")
        with open(path) as fh:
            recs = [json.loads(l) for l in fh if l.strip()]
    finally:
        os.chdir(cwd)
    assert recs, "no E1 trials produced"
    for r in recs:
        assert "direct_path_grade" in r and "direct_path_feasible" in r
        assert r["direct_path_feasible"] == (
            r["direct_path_grade"] <= MAX_GRADE_TAN)
