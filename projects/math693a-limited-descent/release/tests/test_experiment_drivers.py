"""Integration tests for the E3 / E4 trial-record construction.

The full `run_e3` / `run_e4` drivers sweep all 5 terrains x 3 seeds and a
sensitivity sub-experiment, which is far too heavy for a unit test. Instead we
exercise the *exact same* record-building path the drivers use (`_trial` for E3;
the inline record dict for E4 is structurally identical) on a single small
terrain, and check the property that actually matters for the science: the
per-trial record must carry the matching reference (optimum) lengths E5 later
consumes, joined to the SAME (terrain, start).

The headline objective -- "does the rotation heuristic approach the constrained
shortest safe path?" -- only makes sense if every trial's optimality gap is
computed against the optimum for its own start. A leaky or mislabelled join
(start present in trials but dropped from / mismatched against the reference
dict) would silently corrupt every gap without crashing.
"""
import pytest

from src.experiment_e3 import (
    _trial, SEEDS as E3_SEEDS, ROTATION_VARIANTS, SENSITIVITY_TERRAIN,
    SENSITIVITY_DS,
)
from src.experiment_e4 import STRATEGY as E4_STRATEGY
from src.common import prepare_terrain
from src.graph import (
    dijkstra_grade_constrained, reconstruct_path, theta_star_smooth,
    path_length_3d,
)
from src.strategies import STRATEGIES

GRID_N = 28
DS = 0.02
TERRAIN = "T1"


def _setup(seed=0):
    return prepare_terrain(TERRAIN, GRID_N, seed)


def test_trial_carries_matching_reference():
    """Every E3 trial record must carry positive, finite reference lengths that
    match the optimum stored for its own start (no leaky / mismatched join)."""
    setup = _setup(seed=0)
    assert setup.starts, "no reachable starts on the test terrain"
    checked = 0
    for sp in setup.starts:
        rec = _trial(setup, sp, "rotation_cw", DS, "rid", 0, GRID_N,
                     TERRAIN, "main")
        theta = rec["theta_star_length"]
        raw = rec["raw_dijkstra_length"]
        assert theta is not None and raw is not None
        assert theta > 0.0, "non-positive Theta* reference -> undefined gap"
        assert raw >= theta - 1e-9, "raw Dijkstra shorter than Theta* (impossible)"
        # the reference attached must be the one stored for THIS start
        assert tuple(rec["start_ij"]) == sp
        assert rec["theta_star_length"] == pytest.approx(setup.theta_len[sp])
        assert rec["raw_dijkstra_length"] == pytest.approx(setup.raw_len[sp])
        # the result dict fields E5 needs are merged in
        assert "path_length_3d" in rec and "converged" in rec
        assert "n_violations" in rec
        checked += 1
    assert checked > 0


def test_trial_reference_matches_fresh_dijkstra():
    """The stored reference must equal a Theta* optimum rebuilt from scratch --
    catches a reference that is self-consistent but globally wrong (e.g. built
    against the wrong sink or terrain)."""
    setup = _setup(seed=0)
    sp = setup.starts[0]
    rec = _trial(setup, sp, "rotation_ccw", DS, "rid", 0, GRID_N, TERRAIN, "main")
    t = setup.terrain
    dist, prev = dijkstra_grade_constrained(t, sp)
    raw_path = reconstruct_path(prev, sp, t.sink_ij)
    assert raw_path is not None
    raw_len = path_length_3d(t, raw_path)
    theta_len = path_length_3d(t, theta_star_smooth(t, raw_path))
    assert rec["raw_dijkstra_length"] == pytest.approx(raw_len, rel=1e-9)
    assert rec["theta_star_length"] == pytest.approx(theta_len, rel=1e-9)


def test_trial_records_terrain_and_constraint_metadata():
    """A trial must record the terrain it ran on and the grade constraint, so
    E5 can group/filter correctly and the 5-degree limit is auditable."""
    setup = _setup(seed=0)
    rec = _trial(setup, setup.starts[0], "rotation_cw", DS, "rid", 2, GRID_N,
                 TERRAIN, "main")
    assert rec["terrain"] == TERRAIN
    assert rec["seed"] == 2
    assert rec["experiment"] == "E3"
    assert rec["max_grade_degrees"] == 5.0
    # sink must be recorded so endpoints are comparable across methods
    assert tuple(rec["sink_ij"]) == setup.terrain.sink_ij


def test_e3_sweeps_only_rotation_variants():
    """E3 is the rotation heuristic experiment; its declared variant list must
    be exactly the two rotation directions and nothing else, and each must be a
    real strategy key."""
    assert set(ROTATION_VARIANTS) == {"rotation_cw", "rotation_ccw"}
    for v in ROTATION_VARIANTS:
        assert v in STRATEGIES


def test_e4_strategy_is_projection():
    assert E4_STRATEGY == "gradient_projection"
    assert E4_STRATEGY in STRATEGIES


def test_sensitivity_config_sane():
    """The step-size sensitivity sub-experiment must vary ds over distinct
    positive values on a single named terrain (the canonical T2 == Rosenbrock
    ridge)."""
    assert len(set(SENSITIVITY_DS)) == len(SENSITIVITY_DS)
    assert all(d > 0 for d in SENSITIVITY_DS)
    assert SENSITIVITY_TERRAIN == "T2"


def test_seed_sweep_is_not_inert():
    """The drivers sweep SEEDS and re-select starts per seed. If every seed
    produced the same starts the sweep would be a no-op and the seed argument
    silently inert."""
    starts_by_seed = {sd: tuple(_setup(seed=sd).starts) for sd in E3_SEEDS}
    assert len(E3_SEEDS) >= 2
    assert len(set(starts_by_seed.values())) > 1, (
        "seed sweep produced identical starts for every seed")


def test_trial_deterministic_for_fixed_inputs():
    """Same (setup, start, strategy, ds, seed) -> identical record fields."""
    setup = _setup(seed=1)
    sp = setup.starts[0]
    r1 = _trial(setup, sp, "rotation_cw", DS, "rid", 1, GRID_N, TERRAIN, "main")
    r2 = _trial(setup, sp, "rotation_cw", DS, "rid", 1, GRID_N, TERRAIN, "main")
    for key in ("path_length_3d", "n_violations", "converged", "iterations",
                "max_grade", "theta_star_length", "raw_dijkstra_length"):
        assert r1[key] == r2[key], f"non-deterministic field {key}"
