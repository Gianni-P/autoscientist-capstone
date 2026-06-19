"""E2 - unconstrained steepest descent baseline (plan experiment E2).

Run plain steepest descent (no grade check, fixed step ds=0.002) from all
selected start points on all 4 terrains at seed 0. Records 3-D path length,
per-step grade statistics, fraction of steps violating the constraint, and
iterations to convergence. This provides the path-length lower bound and the
constraint-violation upper bound used as a sanity check in E5.

Public API
----------
run_e2(run_id, seed, grid_n, ds) -> dict
"""
import json
import os
import time

import numpy as np

from src.config import (
    GRID_N, MAX_GRADE_DEGREES, MAX_GRADE_TAN,
)
from src.common import prepare_terrain, write_jsonl, ensure_run_dir, RUN_TERRAINS
from src.strategies import run_strategy, DS_DEFAULT, MAX_ITERS


def run_e2(run_id="e2", seed=0, grid_n=GRID_N, ds=DS_DEFAULT):
    out_dir = ensure_run_dir(run_id)
    trial_records = []
    terrain_summaries = []

    for name in RUN_TERRAINS:
        t0 = time.time()
        setup = prepare_terrain(name, grid_n, seed)
        per = []
        for sp in setup.starts:
            res = run_strategy(
                setup.terrain, sp, "unconstrained_steepest_descent",
                ds=ds, max_iters=MAX_ITERS, seed=seed)
            rec = {
                "experiment": "E2",
                "run_id": run_id,
                "seed": seed,
                "grid_n": grid_n,
                "ds": ds,
                "max_grade_degrees": MAX_GRADE_DEGREES,
                "max_grade_tan": MAX_GRADE_TAN,
                "terrain": name,
                "strategy": "unconstrained_steepest_descent",
                "start_ij": list(sp),
                "sink_ij": list(setup.terrain.sink_ij),
                "theta_star_length": setup.theta_len[sp],
                "raw_dijkstra_length": setup.raw_len[sp],
                **res,
            }
            trial_records.append(rec)
            per.append(rec)

        lengths = [r["path_length_3d"] for r in per if r["converged"]]
        terrain_summaries.append({
            "terrain": name,
            "n_starts": len(per),
            "n_converged": int(sum(r["converged"] for r in per)),
            "mean_path_length": float(np.mean(lengths)) if lengths else None,
            "mean_feasibility_rate": float(
                np.mean([r["feasibility_rate"] for r in per])) if per else None,
            "wall_seconds": time.time() - t0,
        })

    write_jsonl(os.path.join(out_dir, "e2_trials.jsonl"), trial_records)
    summary = {
        "experiment": "E2",
        "run_id": run_id,
        "seed": seed,
        "grid_n": grid_n,
        "ds": ds,
        "n_trials": len(trial_records),
        "terrain_summaries": terrain_summaries,
    }
    with open(os.path.join(out_dir, "e2_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
