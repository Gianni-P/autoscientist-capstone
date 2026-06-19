"""Grade-constrained 8-connectivity graph, Dijkstra, and Theta* smoothing.

This module provides the grid-based reference paths for E1:

  * raw 8-connectivity Dijkstra on the grade-constrained weighted graph, where
    each edge weight is the 3-D arc length sqrt(dx^2+dy^2+dz^2) and edges that
    violate the max-grade constraint are removed (weight = inf / absent).
  * Theta*-style line-of-sight post-smoothing of the Dijkstra waypoint path,
    which removes collinear/grid-staircase artefacts and re-measures 3-D
    arc length along any-angle segments.

Public API
----------
path_length_3d(terrain, path_ij) -> float
edge_feasible(terrain, i0, j0, i1, j1) -> bool
dijkstra_grade_constrained(terrain, start_ij) -> (dist_array, prev_dict)
reconstruct_path(prev, start_ij, goal_ij) -> list[(i,j)] or None
theta_star_smooth(terrain, path_ij) -> list[(i,j)]
"""
import heapq
import math

import numpy as np

from src.config import MAX_GRADE_TAN

# 8-connectivity neighbour offsets in grid index space (di, dj)
_NEIGHBORS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def _horizontal_distance(terrain, i0, j0, i1, j1):
    dxw = (j1 - j0) * terrain.dx
    dyw = (i1 - i0) * terrain.dy
    return math.hypot(dxw, dyw)


def _vertical_distance(terrain, i0, j0, i1, j1):
    return abs(terrain.z[i1, j1] - terrain.z[i0, j0])


def edge_feasible(terrain, i0, j0, i1, j1):
    """True if the segment (i0,j0)->(i1,j1) respects the max-grade constraint.

    Grade = |dz| / horizontal_distance. Feasible if <= tan(max_grade).
    A zero-horizontal-distance move is treated as infeasible (degenerate).
    """
    h = _horizontal_distance(terrain, i0, j0, i1, j1)
    if h <= 0.0:
        return False
    dz = _vertical_distance(terrain, i0, j0, i1, j1)
    return (dz / h) <= MAX_GRADE_TAN + 1e-12


def _segment_length_3d(terrain, i0, j0, i1, j1):
    h = _horizontal_distance(terrain, i0, j0, i1, j1)
    dz = _vertical_distance(terrain, i0, j0, i1, j1)
    return math.sqrt(h * h + dz * dz)


def path_length_3d(terrain, path_ij):
    """Total 3-D arc length of a polyline given as a list of (i,j) indices."""
    if path_ij is None or len(path_ij) < 2:
        return 0.0
    total = 0.0
    for (i0, j0), (i1, j1) in zip(path_ij[:-1], path_ij[1:]):
        total += _segment_length_3d(terrain, i0, j0, i1, j1)
    return total


def dijkstra_grade_constrained(terrain, start_ij):
    """Dijkstra over the grade-constrained 8-connectivity grid.

    Returns (dist, prev) where dist is an (n,n) float array of shortest 3-D
    arc-length distances from start (inf if unreachable) and prev is a dict
    mapping (i,j) -> (pi,pj) predecessor.
    """
    n = terrain.n
    dist = np.full((n, n), np.inf, dtype=np.float64)
    prev = {}
    si, sj = start_ij
    dist[si, sj] = 0.0
    pq = [(0.0, si, sj)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if ni < 0 or nj < 0 or ni >= n or nj >= n:
                continue
            if not edge_feasible(terrain, i, j, ni, nj):
                continue
            w = _segment_length_3d(terrain, i, j, ni, nj)
            nd = d + w
            if nd < dist[ni, nj]:
                dist[ni, nj] = nd
                prev[(ni, nj)] = (i, j)
                heapq.heappush(pq, (nd, ni, nj))
    return dist, prev


def reconstruct_path(prev, start_ij, goal_ij):
    """Build the (i,j) path from start to goal using the prev map.

    Returns None if goal is unreachable (no predecessor and goal != start).
    """
    if goal_ij == start_ij:
        return [start_ij]
    if goal_ij not in prev:
        return None
    path = [goal_ij]
    cur = goal_ij
    while cur != start_ij:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def _bresenham(i0, j0, i1, j1):
    """Integer grid cells on the line from (i0,j0) to (i1,j1), inclusive."""
    cells = []
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i1 >= i0 else -1
    sj = 1 if j1 >= j0 else -1
    i, j = i0, j0
    if dj >= di:
        err = dj / 2.0
        while j != j1:
            cells.append((i, j))
            err -= di
            if err < 0:
                i += si
                err += dj
            j += sj
        cells.append((i1, j1))
    else:
        err = di / 2.0
        while i != i1:
            cells.append((i, j))
            err -= dj
            if err < 0:
                j += sj
                err += di
            i += si
        cells.append((i1, j1))
    return cells


def line_of_sight(terrain, i0, j0, i1, j1):
    """Theta*-style any-angle feasibility test between two grid cells.

    The straight segment is feasible if EVERY consecutive pair of cells along
    its Bresenham rasterisation respects the grade constraint. This conserves
    the grade constraint along the smoothed segment.
    """
    cells = _bresenham(i0, j0, i1, j1)
    for (a_i, a_j), (b_i, b_j) in zip(cells[:-1], cells[1:]):
        if not edge_feasible(terrain, a_i, a_j, b_i, b_j):
            return False
    return True


def theta_star_smooth(terrain, path_ij):
    """Line-of-sight smoothing of a Dijkstra path (Theta* post-processing).

    Greedily skips intermediate waypoints whenever a direct feasible
    line-of-sight exists from the last committed waypoint to a later node.
    Returns a new (i,j) polyline whose 3-D length is <= the input by
    construction (straightening can only shorten or preserve length).
    """
    if path_ij is None or len(path_ij) <= 2:
        return list(path_ij) if path_ij is not None else None
    smoothed = [path_ij[0]]
    anchor = 0
    k = 1
    while k < len(path_ij) - 1:
        ai, aj = path_ij[anchor]
        ni, nj = path_ij[k + 1]
        if line_of_sight(terrain, ai, aj, ni, nj):
            k += 1
        else:
            smoothed.append(path_ij[k])
            anchor = k
            k += 1
    smoothed.append(path_ij[-1])
    return smoothed
