"""Fix-pass tests for the grade-constrained grid reference (grid_search.py).

Scope of this review fix pass (per the handoff): correctness of the
Dijkstra / Theta* grade-constrained grid reference, and specifically the newly
added *direct endpoint-to-endpoint grade guard* in ``_line_of_sight``.

These tests are complementary to ``test_supercover_los_consistency.py`` and
``test_grade_constraint.py`` -- they lock in three failure modes those files do
NOT pin down:

  A. The direct-segment grade guard is GENUINELY NECESSARY and ACTIVE. The
     sibling test only asserts the guard *if* it happens to construct a direct
     violation (`if direct_grade > cap: ...`), which silently no-ops if the
     construction fails. Here we build a Z by *direct math* where we PROVE
     (i) every per-cell DDA step is within the cap, yet (ii) the direct grade
     exceeds the cap -- and assert line-of-sight rejects it. If a regression
     dropped the direct guard, the per-cell walk alone would WRONGLY accept the
     segment, so this test would fail. That is the exact T4 n=200 blocker.

  B. The grid reference is DETERMINISTIC. Dijkstra and Theta* use a binary heap
     keyed on float costs; ties can reorder pops. A non-deterministic reference
     optimum makes every downstream optimality_gap noisy. We assert byte-equal
     repeated results (length AND full path) across calls.

  C. Infeasible-by-construction problems are reported as ``(inf, None)`` rather
     than a silently constraint-violating path, and a feasible Theta* path is
     never longer than its Dijkstra counterpart on the SAME (start, goal) at the
     fix-pass resolution.
"""
import math

import numpy as np
import pytest

from src.config import MAX_GRADE_SLOPE, DOMAIN_MIN, DOMAIN_MAX
from src.grid_search import (
    _line_of_sight,
    _supercover_cells,
    dijkstra,
    theta_star,
    build_height_grid,
    select_reachable_start,
)
from src.terrains import get_terrain


def _per_cell_walk_ok(axis, Z, r0, c0, r1, c1):
    """Re-implement ONLY the per-cell DDA walk (no direct guard) the way the
    pre-fix code did, so we can prove the direct guard is what rejects a
    segment. Returns True iff every consecutive crossed-cell step is within
    the grade cap."""
    cells = _supercover_cells(r0, c0, r1, c1)
    for (ra, ca), (rb, cb) in zip(cells[:-1], cells[1:]):
        horiz = math.hypot(axis[cb] - axis[ca], axis[rb] - axis[ra])
        if horiz == 0.0:
            continue
        if abs(Z[rb, cb] - Z[ra, ca]) / horiz > MAX_GRADE_SLOPE:
            return False
    return True


# --------------------------------------------------------------------------- #
# A. The direct-segment grade guard is necessary AND active (T4 n=200 blocker)
# --------------------------------------------------------------------------- #
def test_direct_grade_guard_rejects_what_per_cell_walk_accepts():
    """Construct a segment where the per-cell DDA walk passes but the direct
    endpoint-to-endpoint grade exceeds the cap; the direct guard MUST reject it.

    Geometry: segment (0,0)->(1,2). The round-to-nearest DDA visits a diagonal
    step (0,0)->(1,1) [horiz = sqrt(2)*dx] then an axial step (1,1)->(1,2)
    [horiz = dx]. We give each step a rise at 99% of ITS OWN cap budget. The
    accumulated rise is 0.99*cap*(sqrt(2)+1)*dx over a direct horizontal of
    sqrt(5)*dx, giving a direct grade of 0.99*cap*(sqrt(2)+1)/sqrt(5) ~=
    1.063*cap > cap. So:
        * per-cell walk  -> ACCEPTS (proved below),
        * direct guard   -> REJECTS,
        * _line_of_sight -> must return False (direct guard wins).
    A regression dropping the direct guard would make _line_of_sight return
    True here, failing this test.
    """
    n = 8
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    dx = float(axis[1] - axis[0])
    cap = MAX_GRADE_SLOPE

    cells = _supercover_cells(0, 0, 1, 2)
    # Sanity: DDA really took a diagonal then an axial step (mixed-step case).
    assert cells[0] == (0, 0) and cells[-1] == (1, 2)
    assert len(cells) == 3, f"expected diagonal+axial DDA, got {cells}"

    Z = np.zeros((n, n), dtype=float)
    z = 0.0
    for (ra, ca), (rb, cb) in zip(cells[:-1], cells[1:]):
        horiz = math.hypot(axis[cb] - axis[ca], axis[rb] - axis[ra])
        z += 0.99 * cap * horiz          # each step at 99% of its own cap
        Z[rb, cb] = z

    # (i) Every per-cell DDA step is within the cap.
    assert _per_cell_walk_ok(axis, Z, 0, 0, 1, 2) is True, (
        "test setup wrong: a per-cell step already exceeds the cap"
    )
    # (ii) The DIRECT endpoint-to-endpoint grade exceeds the cap.
    horiz_direct = math.hypot(axis[2] - axis[0], axis[1] - axis[0])
    direct_grade = abs(Z[1, 2] - Z[0, 0]) / horiz_direct
    assert direct_grade > cap, (
        f"test setup wrong: direct grade {direct_grade:.5f} <= cap {cap:.5f}; "
        "the guard would have nothing to catch"
    )
    # (iii) line-of-sight must reject -> proves the direct guard is active.
    assert _line_of_sight(axis, Z, 0, 0, 1, 2) is False, (
        "direct-segment grade guard is missing: line-of-sight approved a "
        f"segment whose direct grade {direct_grade:.5f} exceeds cap {cap:.5f}"
    )


def test_line_of_sight_accepts_when_direct_and_cells_both_ok():
    """Counterpart to the rejection test: a genuinely gentle diagonal where
    BOTH the direct grade and every per-cell step are within the cap must be
    accepted. Guards against an over-eager guard that rejects everything."""
    n = 8
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    dx = float(axis[1] - axis[0])
    cap = MAX_GRADE_SLOPE
    Z = np.zeros((n, n), dtype=float)
    # gentle monotone rise along the diagonal, well under cap per diagonal step
    diag = math.hypot(dx, dx)
    for k in range(4):
        Z[k, k] = k * 0.5 * cap * diag
    assert _per_cell_walk_ok(axis, Z, 0, 0, 3, 3) is True
    horiz_direct = math.hypot(axis[3] - axis[0], axis[3] - axis[0])
    assert abs(Z[3, 3] - Z[0, 0]) / horiz_direct <= cap
    assert _line_of_sight(axis, Z, 0, 0, 3, 3) is True


# --------------------------------------------------------------------------- #
# B. The grid reference is deterministic (no heap-tie / seed nondeterminism)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["T1", "T4"])
def test_dijkstra_deterministic_across_calls(name):
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    len_a, path_a = dijkstra(terrain, n, start, goal)
    len_b, path_b = dijkstra(terrain, n, start, goal)
    assert len_a == len_b, "Dijkstra length is non-deterministic across calls"
    assert path_a == path_b, "Dijkstra path is non-deterministic across calls"


@pytest.mark.parametrize("name", ["T1", "T4"])
def test_theta_star_deterministic_across_calls(name):
    terrain = get_terrain(name)
    n = 60
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    len_a, path_a = theta_star(terrain, n, start, goal)
    len_b, path_b = theta_star(terrain, n, start, goal)
    assert len_a == len_b, "Theta* length is non-deterministic across calls"
    assert path_a == path_b, "Theta* path is non-deterministic across calls"


# --------------------------------------------------------------------------- #
# C. Infeasible reported as (inf, None); Theta* <= Dijkstra at fix-pass scale
# --------------------------------------------------------------------------- #
def test_unreachable_goal_returns_inf_none():
    """If the goal cell is NOT in the sink's reachable set, the reference must
    report (inf, None), never a constraint-violating path. We find such a goal
    directly from the reachable mask so the test is not terrain-luck."""
    from src.grid_search import reachable_from_sink

    terrain = get_terrain("T2")  # steep Rosenbrock: large unreachable region
    n = 40
    sink = terrain.sink
    axis, reachable = reachable_from_sink(terrain, n, sink)
    unreachable = np.argwhere(~reachable)
    if unreachable.size == 0:
        pytest.skip("whole grid reachable at this resolution; no negative case")
    r, c = unreachable[0]
    goal_xy = (float(axis[c]), float(axis[r]))
    length, path = dijkstra(terrain, n, sink, goal_xy)
    assert path is None, "unreachable goal returned a path"
    assert math.isinf(length), "unreachable goal returned a finite length"


def test_theta_star_not_longer_than_dijkstra_at_fixpass_resolution():
    """At the resolution the fix targets, the any-angle reference must not be
    LONGER than the 8-connected one on the same endpoints -- otherwise the
    'optimum' would be mislabeled. Uses a modest n to stay under the time
    budget while still exercising any-angle shortcuts."""
    terrain = get_terrain("T4")
    n = 80
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    dij_len, dpath = dijkstra(terrain, n, start, goal)
    th_len, tpath = theta_star(terrain, n, start, goal)
    assert dpath is not None and tpath is not None
    assert math.isfinite(dij_len) and math.isfinite(th_len)
    assert th_len <= dij_len + 1e-6, (
        f"Theta* len {th_len} exceeds Dijkstra len {dij_len}"
    )


def test_theta_star_reported_path_safe_on_own_heights_n200_T4():
    """The user-visible guarantee of the direct-grade fix at the exact
    resolution (n=200, T4) where the violation manifested: every returned
    straight segment obeys the cap at the z values the path REPORTS, and those
    z equal grid Z (no analytic drift). This is the integration-level proof
    that the unit-level direct guard actually fixes the end-to-end blocker."""
    terrain = get_terrain("T4")
    n = 200
    goal = terrain.sink
    start = select_reachable_start(terrain, n, goal)
    assert start is not None
    th_len, tpath = theta_star(terrain, n, start, goal)
    assert tpath is not None and len(tpath) >= 2
    axis, Z = build_height_grid(terrain, n)
    worst = 0.0
    for (x0, y0, z0), (x1, y1, z1) in zip(tpath[:-1], tpath[1:]):
        horiz = math.hypot(x1 - x0, y1 - y0)
        if horiz > 0:
            worst = max(worst, abs(z1 - z0) / horiz)
    assert worst <= MAX_GRADE_SLOPE + 1e-9, (
        f"returned Theta* segment grade {worst:.5f} exceeds cap "
        f"{MAX_GRADE_SLOPE:.5f} on its own reported heights (n=200 T4)"
    )
    for (x, y, z) in tpath:
        c = int(np.argmin(np.abs(axis - x)))
        r = int(np.argmin(np.abs(axis - y)))
        assert z == pytest.approx(float(Z[r, c]), rel=1e-12, abs=1e-12)
