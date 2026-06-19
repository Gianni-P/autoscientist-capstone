"""Stratified start-point selection for a terrain.

Start points are stratified by Euclidean distance quintile from the sink
(2 points per quintile => N_START_POINTS total), using a fixed seed for
reproducibility. Candidates are restricted to cells that are reachable from
the sink on the grade-constrained feasible graph, so every selected start has
a feasible reference path.

Public API
----------
select_start_points(terrain, dist_from_sink, seed) -> list[(i,j)]
"""
import numpy as np

from src.config import N_START_POINTS, N_QUINTILES, START_SEED


def _euclidean_distance_field(terrain):
    n = terrain.n
    si, sj = terrain.sink_ij
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    dxw = (jj - sj) * terrain.dx
    dyw = (ii - si) * terrain.dy
    return np.sqrt(dxw ** 2 + dyw ** 2)


def select_start_points(terrain, reachable_mask, seed=START_SEED):
    """Pick N_START_POINTS start cells stratified by distance quintile.

    Parameters
    ----------
    terrain : Terrain
    reachable_mask : (n,n) bool array
        True where the cell is reachable from the sink on the feasible graph
        (i.e. finite Dijkstra distance from the sink). The sink cell itself is
        excluded as a start point.
    seed : int

    Returns a list of (i, j) start indices (length <= N_START_POINTS; fewer
    only if some quintile has no reachable candidate).
    """
    rng = np.random.default_rng(seed)
    dist = _euclidean_distance_field(terrain)
    n = terrain.n
    si, sj = terrain.sink_ij

    mask = np.array(reachable_mask, dtype=bool)
    mask[si, sj] = False  # never start at the sink

    cand_flat = np.flatnonzero(mask.ravel())
    if cand_flat.size == 0:
        return []
    cand_dist = dist.ravel()[cand_flat]

    # Distance quintile edges over the candidate set.
    edges = np.quantile(cand_dist, np.linspace(0.0, 1.0, N_QUINTILES + 1))
    per_q = max(1, N_START_POINTS // N_QUINTILES)

    chosen = []
    for q in range(N_QUINTILES):
        lo, hi = edges[q], edges[q + 1]
        if q < N_QUINTILES - 1:
            in_bin = (cand_dist >= lo) & (cand_dist < hi)
        else:
            in_bin = (cand_dist >= lo) & (cand_dist <= hi)
        bin_idx = cand_flat[in_bin]
        if bin_idx.size == 0:
            continue
        take = min(per_q, bin_idx.size)
        picked = rng.choice(bin_idx, size=take, replace=False)
        for flat in picked:
            i, j = divmod(int(flat), n)
            chosen.append((i, j))

    # Deduplicate while preserving order and cap at N_START_POINTS.
    seen = set()
    unique = []
    for c in chosen:
        if c not in seen:
            seen.add(c)
            unique.append(c)
        if len(unique) >= N_START_POINTS:
            break
    return unique
