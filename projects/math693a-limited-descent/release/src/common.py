"""Shared setup helpers reused by experiments E2..E5.

Builds each terrain once on a grid of resolution ``grid_n``, selects the same
stratified start points E1 uses, and computes the Theta*-smoothed
grade-constrained Dijkstra reference path length (the OPTIMUM for the optimality
gap) and the raw Dijkstra length (for the quantisation bias). This reuses the
exact graph/startpoint API from E1 so the reference is byte-for-byte the same
surface and constraint.

Public API
----------
RUN_TERRAINS -> list[str]   (T1..T5 in plan order)
prepare_terrain(name, grid_n, seed) -> TerrainSetup
write_jsonl(path, records) -> None
ensure_run_dir(run_id) -> str
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

from src.config import RUNS_DIR
from src.terrains import build_terrain, list_terrains
from src.graph import (
    dijkstra_grade_constrained, reconstruct_path, theta_star_smooth,
    path_length_3d,
)
from src.startpoints import select_start_points

RUN_TERRAINS = list_terrains()


@dataclass
class TerrainSetup:
    name: str
    terrain: object                   # GridTerrain (grid API consumers expect)
    starts: list                      # list of (i,j)
    raw_len: dict = field(default_factory=dict)     # (i,j) -> raw dijkstra 3d len
    theta_len: dict = field(default_factory=dict)   # (i,j) -> theta* 3d len
    reachable_cells: int = 0


def prepare_terrain(name, grid_n, seed):
    """Build a gridded terrain, select start points, compute reference lengths.

    Returns a TerrainSetup whose ``terrain`` is a ``GridTerrain`` sampled at
    resolution ``grid_n`` (so every downstream consumer gets the
    .n/.dx/.dy/.z/.xs/.ys/.sink_ij/.sink_xy grid API). For each start the raw
    Dijkstra and Theta*-smoothed grade-constrained reference 3-D lengths are
    stored (the optimum). Starts for which the sink is unreachable are dropped.
    """
    terrain = build_terrain(name, grid_n)
    sink_dist, _ = dijkstra_grade_constrained(terrain, terrain.sink_ij)
    reachable = np.isfinite(sink_dist)
    starts = select_start_points(terrain, reachable, seed=seed)

    setup = TerrainSetup(
        name=name, terrain=terrain, starts=[],
        reachable_cells=int(np.count_nonzero(reachable)),
    )
    for sp in starts:
        dist, prev = dijkstra_grade_constrained(terrain, sp)
        raw_path = reconstruct_path(prev, sp, terrain.sink_ij)
        if raw_path is None:
            continue
        raw_len = path_length_3d(terrain, raw_path)
        smoothed = theta_star_smooth(terrain, raw_path)
        theta_len = path_length_3d(terrain, smoothed)
        setup.starts.append(sp)
        setup.raw_len[sp] = float(raw_len)
        setup.theta_len[sp] = float(theta_len)
    return setup


def write_jsonl(path, records):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def ensure_run_dir(run_id):
    out_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
