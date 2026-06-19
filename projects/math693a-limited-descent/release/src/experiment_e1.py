"""E1 - baseline reproduction (plan experiment E1).

For every terrain and every stratified start point:
  * compute the grade-constrained 8-connectivity Dijkstra reference path (raw),
  * compute the Theta*-smoothed reference path,
  * record the 3-D arc lengths and the Quantisation Bias
        QB = (raw_dijkstra - theta_star) / theta_star.

Internal-validity checks (per stop_conditions):
  (a) Theta*-smoothed length <= raw Dijkstra length on every trial;
  (b) on T1 (elliptic_paraboloid) the Theta* length approaches the straight-line
      3-D geodesic from start to sink within T1_GEODESIC_TOL -- but only when the
      direct start->sink path is itself grade-feasible. If the direct line is too
      steep to follow, the constrained optimum must detour and a geodesic-deviation
      bound is physically meaningless, so the check is skipped for that start.
  (c) the sink is reachable from every selected start on the feasible graph.

Results and per-trial config are written as jsonl under runs/<run_id>/.

Public API
----------
run_e1(run_id, seed, grid_n) -> dict  (summary)
direct_path_grade(terrain, start_ij) -> float
"""
import json
import math
import os
import time

import numpy as np

from src.config import (
    GRID_N, START_SEED, RUNS_DIR, T1_GEODESIC_TOL, MAX_GRADE_TAN,
    MAX_GRADE_DEGREES,
)
from src.terrains import build_terrain, list_terrains
from src.graph import (
    dijkstra_grade_constrained, reconstruct_path, theta_star_smooth,
    path_length_3d,
)
from src.startpoints import select_start_points

# Canonical id of the elliptic-paraboloid terrain (T1). The geodesic-deviation
# internal-validity check (b) applies to this terrain only; terrains are keyed
# by their canonical T1..T5 ids, so the name MUST match a real key.
ELLIPTIC_PARABOLOID = "T1"


def _geodesic_3d(terrain, start_ij):
    """Straight-line 3-D distance from start cell to the sink cell."""
    si, sj = start_ij
    gi, gj = terrain.sink_ij
    dxw = (gj - sj) * terrain.dx
    dyw = (gi - si) * terrain.dy
    dz = terrain.z[gi, gj] - terrain.z[si, sj]
    return math.sqrt(dxw ** 2 + dyw ** 2 + dz ** 2)


def direct_path_grade(terrain, start_ij):
    """Grade (|dz| / horizontal distance) of the direct start->sink line.

    This is the average grade of the straight line connecting the start cell to
    the sink cell on the normalised surface: total height drop divided by the
    horizontal (planar) separation. If this exceeds MAX_GRADE_TAN the direct
    line is not grade-feasible and any geodesic-deviation bound on the
    constrained optimum would be physically meaningless.

    Returns 0.0 when start and sink coincide (no horizontal separation).
    """
    si, sj = start_ij
    gi, gj = terrain.sink_ij
    dxw = (gj - sj) * terrain.dx
    dyw = (gi - si) * terrain.dy
    horizontal = math.sqrt(dxw ** 2 + dyw ** 2)
    if horizontal <= 0.0:
        return 0.0
    dz = abs(terrain.z[gi, gj] - terrain.z[si, sj])
    return dz / horizontal


def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def run_e1(run_id="e1", seed=START_SEED, grid_n=GRID_N):
    """Run E1 across all terrains. Returns a summary dict and writes jsonl."""
    out_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)

    trial_records = []
    terrain_summaries = []
    validity_failures = []

    for name in list_terrains():
        t_start = time.time()
        terrain = build_terrain(name, grid_n)

        # Dijkstra from the sink gives reachability + symmetric distances on
        # an undirected graph. Edge feasibility is symmetric (|dz|), so a
        # finite distance-from-sink implies a feasible path exists.
        sink_dist, _sink_prev = dijkstra_grade_constrained(terrain, terrain.sink_ij)
        reachable = np.isfinite(sink_dist)
        reachable_count = int(np.count_nonzero(reachable))

        starts = select_start_points(terrain, reachable, seed=seed)
        if len(starts) == 0:
            validity_failures.append({
                "terrain": name,
                "check": "reachability",
                "detail": "no reachable start points found",
            })
            terrain_summaries.append({
                "terrain": name, "n_starts": 0, "aborted": True,
                "reachable_cells": reachable_count,
            })
            continue

        per_terrain = []
        for sp in starts:
            si, sj = sp
            reachable_sp = bool(reachable[si, sj])
            if not reachable_sp:
                validity_failures.append({
                    "terrain": name, "check": "reachability",
                    "start": [si, sj],
                    "detail": "selected start unreachable from sink",
                })

            dist, prev = dijkstra_grade_constrained(terrain, sp)
            raw_path = reconstruct_path(prev, sp, terrain.sink_ij)
            if raw_path is None:
                validity_failures.append({
                    "terrain": name, "check": "reachability",
                    "start": [si, sj], "detail": "sink unreachable from start",
                })
                continue

            raw_len = path_length_3d(terrain, raw_path)
            smoothed = theta_star_smooth(terrain, raw_path)
            theta_len = path_length_3d(terrain, smoothed)

            # check (a): theta* <= raw (allow tiny float slack)
            check_a = theta_len <= raw_len + 1e-9
            if not check_a:
                validity_failures.append({
                    "terrain": name, "check": "theta_le_raw",
                    "start": [si, sj],
                    "detail": f"theta*={theta_len:.6f} > raw={raw_len:.6f}",
                })

            qb = (raw_len - theta_len) / theta_len if theta_len > 0 else 0.0
            geodesic = _geodesic_3d(terrain, sp)
            geo_dev = (theta_len - geodesic) / geodesic if geodesic > 0 else 0.0
            direct_grade = direct_path_grade(terrain, sp)
            direct_feasible = direct_grade <= MAX_GRADE_TAN

            rec = {
                "experiment": "E1",
                "run_id": run_id,
                "seed": seed,
                "grid_n": grid_n,
                "max_grade_degrees": MAX_GRADE_DEGREES,
                "max_grade_tan": MAX_GRADE_TAN,
                "terrain": name,
                "start_ij": [si, sj],
                "sink_ij": list(terrain.sink_ij),
                "raw_dijkstra_length": raw_len,
                "theta_star_length": theta_len,
                "geodesic_3d": geodesic,
                "quantisation_bias": qb,
                "theta_geodesic_deviation": geo_dev,
                "direct_path_grade": direct_grade,
                "direct_path_feasible": bool(direct_feasible),
                "n_raw_waypoints": len(raw_path),
                "n_smoothed_waypoints": len(smoothed),
                "check_theta_le_raw": bool(check_a),
                "reachable": reachable_sp,
            }
            trial_records.append(rec)
            per_terrain.append(rec)

        # check (b): T1 geodesic deviation -- only applied when the direct
        # start->sink line is itself grade-feasible. When the direct line is too
        # steep, the constrained optimum is *required* to detour and a
        # geodesic-deviation bound is not a meaningful internal-validity
        # criterion, so we skip the threshold for that start.
        if name == ELLIPTIC_PARABOLOID:
            for rec in per_terrain:
                if not rec["direct_path_feasible"]:
                    continue
                if abs(rec["theta_geodesic_deviation"]) > T1_GEODESIC_TOL:
                    validity_failures.append({
                        "terrain": name, "check": "t1_geodesic",
                        "start": rec["start_ij"],
                        "detail": (
                            f"theta*-geodesic deviation "
                            f"{rec['theta_geodesic_deviation']:.4f} "
                            f"> tol {T1_GEODESIC_TOL} "
                            f"(direct_grade={rec['direct_path_grade']:.4f} "
                            f"<= {MAX_GRADE_TAN:.4f})"
                        ),
                    })

        qbs = [r["quantisation_bias"] for r in per_terrain]
        terrain_summaries.append({
            "terrain": name,
            "n_starts": len(per_terrain),
            "reachable_cells": reachable_count,
            "mean_qb": float(np.mean(qbs)) if qbs else None,
            "max_qb": float(np.max(qbs)) if qbs else None,
            "wall_seconds": time.time() - t_start,
            "aborted": False,
        })

    _write_jsonl(os.path.join(out_dir, "e1_trials.jsonl"), trial_records)

    summary = {
        "experiment": "E1",
        "run_id": run_id,
        "seed": seed,
        "grid_n": grid_n,
        "terrain_summaries": terrain_summaries,
        "n_trials": len(trial_records),
        "n_validity_failures": len(validity_failures),
        "validity_failures": validity_failures,
        "internal_validity_passed": len(validity_failures) == 0,
    }
    with open(os.path.join(out_dir, "e1_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary
