"""E3 - rotation heuristic, CW and CCW (plan experiment E3).

Run the clockwise (CW) and counter-clockwise (CCW) rotation heuristics
(1-degree angular increment, 360-degree bound before declaring failure,
fixed step ds=0.002) from all selected start points on all 5 terrains at
seeds 0, 1, 2. Also runs the step-size sensitivity sub-experiment on T2
(rosenbrock_ridge) with ds in {0.001, 0.002, 0.005}.

Records per-step grade, 3-D path length, iterations, and convergence
success/failure for every trial.

Public API
----------
SEEDS, SENSITIVITY_DS, SENSITIVITY_TERRAIN
run_e3(run_id, seed, grid_n, ds) -> dict
"""
import json
import os
import time

import numpy as np

from src.config import GRID_N, MAX_GRADE_DEGREES, MAX_GRADE_TAN
from src.common import prepare_terrain, write_jsonl, ensure_run_dir, RUN_TERRAINS
from src.strategies import run_strategy, DS_DEFAULT, MAX_ITERS

SEEDS = [0, 1, 2]
SENSITIVITY_DS = [0.001, 0.002, 0.005]
# Canonical terrain id (T2 == rosenbrock_ridge). MUST match the keys the
# terrains module exposes, otherwise prepare_terrain raises KeyError.
SENSITIVITY_TERRAIN = "T2"
ROTATION_VARIANTS = ["rotation_cw", "rotation_ccw"]


def _trial(setup, sp, strategy, ds, run_id, seed, grid_n, name, sub):
    res = run_strategy(setup.terrain, sp, strategy, ds=ds,
                       max_iters=MAX_ITERS, seed=seed)
    return {
        "experiment": "E3",
        "run_id": run_id,
        "seed": seed,
        "grid_n": grid_n,
        "ds": ds,
        "subexperiment": sub,
        "max_grade_degrees": MAX_GRADE_DEGREES,
        "max_grade_tan": MAX_GRADE_TAN,
        "terrain": name,
        "strategy": strategy,
        "start_ij": list(sp),
        "sink_ij": list(setup.terrain.sink_ij),
        "theta_star_length": setup.theta_len[sp],
        "raw_dijkstra_length": setup.raw_len[sp],
        **res,
    }


def run_e3(run_id="e3", seed=0, grid_n=GRID_N, ds=DS_DEFAULT):
    """`seed` is accepted for CLI symmetry but E3 sweeps SEEDS internally."""
    out_dir = ensure_run_dir(run_id)
    trial_records = []
    terrain_summaries = []

    # Cache terrain setups per (terrain, seed) -- start selection depends on seed.
    setups = {}

    def get_setup(name, sd):
        key = (name, sd)
        if key not in setups:
            setups[key] = prepare_terrain(name, grid_n, sd)
        return setups[key]

    # --- main sweep: all terrains x both rotation variants x SEEDS ---
    for name in RUN_TERRAINS:
        t0 = time.time()
        per = []
        for sd in SEEDS:
            setup = get_setup(name, sd)
            for sp in setup.starts:
                for variant in ROTATION_VARIANTS:
                    rec = _trial(setup, sp, variant, ds, run_id, sd,
                                 grid_n, name, "main")
                    trial_records.append(rec)
                    per.append(rec)
        lengths = [r["path_length_3d"] for r in per if r["converged"]]
        terrain_summaries.append({
            "terrain": name,
            "n_trials": len(per),
            "n_converged": int(sum(r["converged"] for r in per)),
            "mean_path_length": float(np.mean(lengths)) if lengths else None,
            "wall_seconds": time.time() - t0,
        })

    # --- step-size sensitivity sub-experiment on T2 only ---
    sens_records = []
    for sd in SEEDS:
        setup = get_setup(SENSITIVITY_TERRAIN, sd)
        for sp in setup.starts:
            for variant in ROTATION_VARIANTS:
                for sds in SENSITIVITY_DS:
                    rec = _trial(setup, sp, variant, sds, run_id, sd,
                                 grid_n, SENSITIVITY_TERRAIN, "step_size")
                    sens_records.append(rec)
    trial_records.extend(sens_records)

    write_jsonl(os.path.join(out_dir, "e3_trials.jsonl"), trial_records)
    summary = {
        "experiment": "E3",
        "run_id": run_id,
        "seeds": SEEDS,
        "grid_n": grid_n,
        "ds": ds,
        "sensitivity_ds": SENSITIVITY_DS,
        "sensitivity_terrain": SENSITIVITY_TERRAIN,
        "n_trials": len(trial_records),
        "n_main_trials": len(trial_records) - len(sens_records),
        "n_sensitivity_trials": len(sens_records),
        "terrain_summaries": terrain_summaries,
    }
    with open(os.path.join(out_dir, "e3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
