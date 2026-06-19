"""Grade-constrained grid path planning: Dijkstra (8-connected) and Theta*.

We discretize a terrain on an N x N grid over [-2, 2]^2. Each node has a 3-D
position (x, y, z=height). An 8-connected edge is *walkable* only if its grade
|dz| / horizontal_distance <= MAX_GRADE_SLOPE. Edge cost is 3-D Euclidean
length. Dijkstra finds the shortest grade-constrained path. Theta* applies
line-of-sight (any-angle) corrections to remove grid angular quantization bias.

Both return (path_length, path_xyz) where path_xyz is a list of (x,y,z) nodes,
or (inf, None) if no feasible path exists.

Height-source consistency
--------------------------
Both the grade-feasibility test (_line_of_sight) and the path cost
(_seg_length_3d) read the SAME grid-snapped height array Z. An earlier version
checked feasibility against the analytic terrain.height() while costing against
grid Z, so Theta* could approve a segment whose grid-snapped node heights
actually violated the grade cap. _line_of_sight now walks the grid cells the
segment crosses (Bresenham DDA cell walk) and checks the grade between
consecutive crossed cells using Z, exactly the heights the returned path
reports.

Direct-segment grade guard
--------------------------
A subtle gap remained: the per-cell DDA walk mixes diagonal (horizontal =
sqrt(2)*dx) and axial (horizontal = dx) steps. Each step could individually
sit just under the cap while the DIRECT endpoint-to-endpoint grade exceeds it,
because the accumulated |dz| can reach cap*(sqrt(2)+1)*dx over a direct
distance of only sqrt(5)*dx (about 1.08*cap). Since the returned path reports
straight segments between Theta* waypoints, the user-visible grade is the
direct endpoint-to-endpoint grade. _line_of_sight therefore also checks the
direct grade between (r0,c0) and (r1,c1) up front and rejects any segment whose
direct grade exceeds the cap.
"""
import heapq
import math

import numpy as np

from src.config import DOMAIN_MIN, DOMAIN_MAX, MAX_GRADE_SLOPE


def build_height_grid(terrain, n):
    """Return (axis, Z) where axis is length-n coords and Z is n x n heights."""
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    gx, gy = np.meshgrid(axis, axis)
    Z = np.asarray(terrain.height(gx, gy), dtype=float)
    return axis, Z


def _nearest_index(axis, coord):
    return int(np.argmin(np.abs(axis - coord)))


# 8-connected neighbor offsets
_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]


def _edge_ok_and_cost(axis, Z, r0, c0, r1, c1):
    """Return 3-D cost if edge (r0,c0)->(r1,c1) is grade-feasible else None."""
    dx = axis[c1] - axis[c0]
    dy = axis[r1] - axis[r0]
    horiz = math.hypot(dx, dy)
    if horiz == 0.0:
        return None
    dz = Z[r1, c1] - Z[r0, c0]
    if abs(dz) / horiz > MAX_GRADE_SLOPE:
        return None
    return math.sqrt(horiz * horiz + dz * dz)


def dijkstra(terrain, n, start_xy, goal_xy):
    """Grade-constrained 8-connected Dijkstra. Returns (length, path_xyz)."""
    axis, Z = build_height_grid(terrain, n)
    sr = _nearest_index(axis, start_xy[1])
    sc = _nearest_index(axis, start_xy[0])
    gr = _nearest_index(axis, goal_xy[1])
    gc = _nearest_index(axis, goal_xy[0])

    dist = np.full((n, n), np.inf)
    prev = {}
    dist[sr, sc] = 0.0
    pq = [(0.0, sr, sc)]
    visited = np.zeros((n, n), dtype=bool)

    while pq:
        d, r, c = heapq.heappop(pq)
        if visited[r, c]:
            continue
        visited[r, c] = True
        if (r, c) == (gr, gc):
            break
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n or visited[nr, nc]:
                continue
            cost = _edge_ok_and_cost(axis, Z, r, c, nr, nc)
            if cost is None:
                continue
            nd = d + cost
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[(nr, nc)] = (r, c)
                heapq.heappush(pq, (nd, nr, nc))

    if not np.isfinite(dist[gr, gc]):
        return math.inf, None

    # Reconstruct
    path = []
    cur = (gr, gc)
    while cur != (sr, sc):
        r, c = cur
        path.append((axis[c], axis[r], float(Z[r, c])))
        cur = prev[cur]
    path.append((axis[sc], axis[sr], float(Z[sr, sc])))
    path.reverse()
    return float(dist[gr, gc]), path


def reachable_from_sink(terrain, n, sink_xy):
    """Return (axis, reachable_mask) of cells reachable from the sink.

    A cell is reachable iff there exists a grade-constrained 8-connected path
    from the sink node to it. This is a single multi-source-free Dijkstra/BFS
    flood from the sink. Used to pick start points that have a feasible
    reference path (see select_reachable_start)."""
    axis, Z = build_height_grid(terrain, n)
    sr = _nearest_index(axis, sink_xy[1])
    sc = _nearest_index(axis, sink_xy[0])

    reachable = np.zeros((n, n), dtype=bool)
    reachable[sr, sc] = True
    stack = [(sr, sc)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n or reachable[nr, nc]:
                continue
            if _edge_ok_and_cost(axis, Z, r, c, nr, nc) is None:
                continue
            reachable[nr, nc] = True
            stack.append((nr, nc))
    return axis, reachable


def select_reachable_start(terrain, n, sink_xy, prefer_far=True):
    """Return an (x, y) start that is grade-feasibly reachable from the sink.

    Picks, among all cells reachable from the sink (excluding the sink itself),
    the one whose Euclidean distance from the sink is largest (prefer_far) so
    the resulting path is non-trivial. Returns None if no non-sink cell is
    reachable (degenerate terrain)."""
    axis, reachable = reachable_from_sink(terrain, n, sink_xy)
    sr = _nearest_index(axis, sink_xy[1])
    sc = _nearest_index(axis, sink_xy[0])
    mask = reachable.copy()
    mask[sr, sc] = False
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        return None
    sink_x, sink_y = axis[sc], axis[sr]
    best = None
    best_d = -1.0 if prefer_far else math.inf
    for r, c in idxs:
        d = math.hypot(axis[c] - sink_x, axis[r] - sink_y)
        if (prefer_far and d > best_d) or (not prefer_far and d < best_d):
            best_d = d
            best = (float(axis[c]), float(axis[r]))
    return best


def _supercover_cells(r0, c0, r1, c1):
    """Return the ordered list of grid cells a straight segment crosses.

    Uses an integer DDA (round-to-nearest Bresenham) so the returned sequence
    of (row, col) cells covers the segment from (r0,c0) to (r1,c1), starting at
    (r0,c0) and ending at (r1,c1). The cap-correctness guarantee for
    line-of-sight does NOT rely on this being a *true* supercover (which would
    emit both cells at exact corner crossings): the direct-segment grade guard
    in _line_of_sight independently bounds the endpoint-to-endpoint grade, so
    no skipped corner cell can hide a cap violation between the segment's
    reported endpoints.
    """
    cells = []
    dr = r1 - r0
    dc = c1 - c0
    n_steps = max(abs(dr), abs(dc))
    if n_steps == 0:
        return [(r0, c0)]
    for k in range(n_steps + 1):
        t = k / n_steps
        rr = int(round(r0 + t * dr))
        cc = int(round(c0 + t * dc))
        if not cells or cells[-1] != (rr, cc):
            cells.append((rr, cc))
    return cells


def _line_of_sight(axis, Z, r0, c0, r1, c1):
    """True if the straight segment (r0,c0)->(r1,c1) is grade-feasible.

    Heights are read from the GRID array Z (not analytic terrain.height) so the
    feasibility test is consistent with _seg_length_3d and with the heights the
    returned path actually reports.

    Two checks must BOTH pass:

      1. Direct-segment guard: the endpoint-to-endpoint grade
         |Z[r1,c1]-Z[r0,c0]| / horiz_direct must be within MAX_GRADE_SLOPE.
         This is the grade the user sees on the returned straight segment, and
         it is NOT implied by the per-cell DDA checks (mixed diagonal/axial
         steps can each pass while the direct grade exceeds the cap).

      2. Per-cell walk: the grade between consecutive crossed cells must also
         stay within MAX_GRADE_SLOPE, so no intermediate crossed cell is a
         steep spike the straight-line endpoints happen to average out.
    """
    if (r0, c0) == (r1, c1):
        return False
    # 1. Direct endpoint-to-endpoint grade guard.
    horiz_direct = math.hypot(axis[c1] - axis[c0], axis[r1] - axis[r0])
    if horiz_direct == 0.0:
        return False
    if abs(Z[r1, c1] - Z[r0, c0]) / horiz_direct > MAX_GRADE_SLOPE:
        return False
    # 2. Per-cell DDA walk.
    cells = _supercover_cells(r0, c0, r1, c1)
    for (ra, ca), (rb, cb) in zip(cells[:-1], cells[1:]):
        horiz = math.hypot(axis[cb] - axis[ca], axis[rb] - axis[ra])
        if horiz == 0.0:
            continue
        dz = Z[rb, cb] - Z[ra, ca]
        if abs(dz) / horiz > MAX_GRADE_SLOPE:
            return False
    return True


def _seg_length_3d(axis, Z, r0, c0, r1, c1):
    x0, y0 = axis[c0], axis[r0]
    x1, y1 = axis[c1], axis[r1]
    horiz = math.hypot(x1 - x0, y1 - y0)
    dz = Z[r1, c1] - Z[r0, c0]
    return math.sqrt(horiz * horiz + dz * dz)


def theta_star(terrain, n, start_xy, goal_xy):
    """Any-angle Theta* on the grade-constrained grid. Returns (length, path)."""
    axis, Z = build_height_grid(terrain, n)
    sr = _nearest_index(axis, start_xy[1])
    sc = _nearest_index(axis, start_xy[0])
    gr = _nearest_index(axis, goal_xy[1])
    gc = _nearest_index(axis, goal_xy[0])

    g_score = np.full((n, n), np.inf)
    parent = {}
    g_score[sr, sc] = 0.0
    parent[(sr, sc)] = (sr, sc)

    def heur(r, c):
        return math.hypot(axis[c] - axis[gc], axis[r] - axis[gr])

    pq = [(heur(sr, sc), sr, sc)]
    closed = np.zeros((n, n), dtype=bool)

    while pq:
        _, r, c = heapq.heappop(pq)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == (gr, gc):
            break
        pr, pc = parent[(r, c)]
        for dr, dc in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= n or nc < 0 or nc >= n or closed[nr, nc]:
                continue
            base_cost = _edge_ok_and_cost(axis, Z, r, c, nr, nc)
            if base_cost is None:
                continue
            # Path 2: try to connect parent(r,c) directly to neighbor, using the
            # SAME grid Z as the cost (consistent feasibility/cost).
            if _line_of_sight(axis, Z, pr, pc, nr, nc):
                cand_parent = (pr, pc)
                cand_g = g_score[pr, pc] + _seg_length_3d(axis, Z, pr, pc, nr, nc)
            else:
                cand_parent = (r, c)
                cand_g = g_score[r, c] + base_cost
            if cand_g < g_score[nr, nc]:
                g_score[nr, nc] = cand_g
                parent[(nr, nc)] = cand_parent
                heapq.heappush(pq, (cand_g + heur(nr, nc), nr, nc))

    if not np.isfinite(g_score[gr, gc]):
        return math.inf, None

    path = []
    cur = (gr, gc)
    while True:
        r, c = cur
        path.append((axis[c], axis[r], float(Z[r, c])))
        if cur == (sr, sc):
            break
        cur = parent[cur]
    path.reverse()
    return float(g_score[gr, gc]), path
