"""Grid-search height-source-consistency pitfalls (review fix pass, grid_search.py).

The blocker fixed in this pass: ``_line_of_sight`` used to test grade
feasibility against the *analytic* ``terrain.height`` while the path COST
(``_seg_length_3d``) and the returned node heights came from the grid-snapped
array ``Z``. On a discretised surface those two height sources disagree, so
Theta* could approve an any-angle shortcut whose grid-snapped endpoints
actually violate the 5-degree grade cap -- handing back a "constrained shortest
safe path" reference that is not actually safe. The fix re-derives line-of-sight
from the supercover cell walk reading ``Z`` AND adds a direct
endpoint-to-endpoint grade guard (the mixed diagonal/axial DDA steps could each
pass while the direct grade exceeds the cap).

These tests lock in:

  1. ``_supercover_cells`` contract -- the new helper Theta* relies on. The
     returned cell sequence must START at (r0,c0), END at (r1,c1), be free of
     consecutive duplicates, and step by at most one row/col between cells (so
     no crossed cell is skipped -- a skipped cell could hide a steep step). The
     length is asserted only as a LOWER bound (a true supercover may emit more
     cells at corner crossings); contiguity and endpoints are the
     safety-critical properties.

  2. ``_line_of_sight`` reads GRID Z, not analytic height. We pass synthetic Z
     arrays where the feasibility decision must follow Z entries: a step that is
     grade-INFEASIBLE in Z must be rejected, and flipping only a Z entry must
     flip the decision. A regression to the old analytic-height check would
     break these (Z is decoupled from any terrain here).

  3. End-to-end: whatever path Theta* RETURNS is grade-feasible when measured on
     the SAME heights the path reports, and those reported heights equal grid Z
     at each node (no analytic drift). This is the user-visible guarantee of the
     height-source fix. We exercise n=200, the resolution at which the
     direct-segment grade gap on T4 manifested before the fix.
"""
import math

import numpy as np
import pytest

from src.config import MAX_GRADE_SLOPE, DOMAIN_MIN, DOMAIN_MAX
from src.grid_search import (
    _supercover_cells, _line_of_sight, theta_star, build_height_grid,
)
from src.terrains import get_terrain


# --------------------------------------------------------------------------- #
# 1. _supercover_cells contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r0,c0,r1,c1", [
    (0, 0, 0, 0),      # degenerate (same cell)
    (0, 0, 0, 5),      # horizontal
    (0, 0, 5, 0),      # vertical
    (0, 0, 5, 5),      # pure diagonal
    (2, 1, 9, 4),      # shallow slope
    (1, 2, 4, 9),      # the transpose
    (9, 4, 2, 1),      # reversed direction
])
def test_supercover_endpoints_and_contiguity(r0, c0, r1, c1):
    cells = _supercover_cells(r0, c0, r1, c1)
    assert cells[0] == (r0, c0), "supercover must start at the segment start"
    assert cells[-1] == (r1, c1), "supercover must end at the segment end"
    # no consecutive duplicates
    for a, b in zip(cells[:-1], cells[1:]):
        assert a != b, "consecutive duplicate cells in supercover walk"
        # each step advances by at most one in each axis (no cell skipped)
        assert abs(b[0] - a[0]) <= 1 and abs(b[1] - a[1]) <= 1, (
            f"supercover skipped a cell stepping {a}->{b}"
        )
    # The number of cells must at least cover the Chebyshev distance + 1. A true
    # supercover may emit MORE cells at exact corner crossings, so this is a
    # lower bound only -- contiguity + endpoints are the safety-critical checks.
    min_len = max(abs(r1 - r0), abs(c1 - c0)) + 1
    assert len(cells) >= min_len


def test_supercover_diagonal_is_monotone():
    """A diagonal walk visits each (k,k); it must not wander off-diagonal."""
    cells = _supercover_cells(0, 0, 6, 6)
    assert cells == [(k, k) for k in range(7)]


# --------------------------------------------------------------------------- #
# 2. _line_of_sight reads grid Z, not analytic height
# --------------------------------------------------------------------------- #
def test_line_of_sight_uses_grid_Z_not_analytic_height():
    """Feasibility must follow the grid array Z passed in.

    Build a Z where the step between two adjacent cells is just OVER the grade
    cap; line-of-sight must reject it. Then make every consecutive step just
    UNDER the cap; line-of-sight must accept it. The decision must depend ONLY
    on Z (no hidden analytic-height lookup).
    """
    n = 6
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    dx = float(axis[1] - axis[0])
    over = (MAX_GRADE_SLOPE + 0.05) * dx
    under = (MAX_GRADE_SLOPE - 0.02) * dx

    Z_bad = np.zeros((n, n), dtype=float)
    Z_bad[0, 1] = over          # the (0,0)->(0,1) step exceeds the cap
    assert _line_of_sight(axis, Z_bad, 0, 0, 0, 2) is False, (
        "line-of-sight approved a segment whose grid step exceeds the grade cap"
    )

    Z_ok = np.zeros((n, n), dtype=float)
    Z_ok[0, 1] = under
    Z_ok[0, 2] = 2 * under      # each consecutive step stays within cap
    assert _line_of_sight(axis, Z_ok, 0, 0, 0, 2) is True, (
        "line-of-sight rejected a segment that is grade-feasible on grid Z"
    )


def test_line_of_sight_decision_flips_with_Z_only():
    """Only Z entries matter: change one crossed-cell height and the decision
    must flip. This is the exact bug class the fix addresses (the decision must
    not consult analytic height)."""
    n = 8
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    dx = float(axis[1] - axis[0])
    Z = np.zeros((n, n), dtype=float)
    # flat diagonal first -> feasible
    assert _line_of_sight(axis, Z, 0, 0, 3, 3) is True
    # plant a steep step on a crossed diagonal cell -> must become infeasible
    Z[2, 2] = (MAX_GRADE_SLOPE + 0.2) * math.hypot(dx, dx)
    assert _line_of_sight(axis, Z, 0, 0, 3, 3) is False


def test_line_of_sight_direct_grade_guard():
    """A segment whose per-cell DDA steps each pass but whose DIRECT
    endpoint-to-endpoint grade exceeds the cap must be rejected.

    Construct a knight's-move segment (r0,c0)->(r0+1,c0+2): the DDA visits a
    diagonal step then an axial step. We pick a dz that keeps each DDA step just
    under the cap but pushes the direct grade just over it -- exactly the gap
    the direct-segment guard closes.
    """
    n = 8
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    dx = float(axis[1] - axis[0])
    cap = MAX_GRADE_SLOPE
    Z = np.zeros((n, n), dtype=float)
    # Segment (0,0)->(1,2). DDA cells: (0,0),(0,1) or (1,1)... use round-to-near.
    # Place intermediate + endpoint heights so each DDA step <= cap but the
    # direct grade > cap.
    cells = _supercover_cells(0, 0, 1, 2)
    # Assign a small per-step rise that stays under each step's cap budget but
    # accumulates beyond the direct budget.
    z = 0.0
    for (ra, ca), (rb, cb) in zip(cells[:-1], cells[1:]):
        horiz = math.hypot(axis[cb] - axis[ca], axis[rb] - axis[ra])
        z += 0.99 * cap * horiz          # each step at 99% of its own cap
        Z[rb, cb] = z
    horiz_direct = math.hypot(axis[2] - axis[0], axis[1] - axis[0])
    direct_grade = abs(Z[1, 2] - Z[0, 0]) / horiz_direct
    # Only meaningful if we actually built a direct violation.
    if direct_grade > cap:
        assert _line_of_sight(axis, Z, 0, 0, 1, 2) is False, (
            f"direct grade {direct_grade:.4f} exceeds cap {cap:.4f} but "
            "line-of-sight approved the segment"
        )


def test_line_of_sight_same_cell_is_false():
    n = 4
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    Z = np.zeros((n, n), dtype=float)
    assert _line_of_sight(axis, Z, 1, 1, 1, 1) is False


# --------------------------------------------------------------------------- #
# 3. End-to-end: returned Theta* path is feasible on its OWN reported heights
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,start", [("T1", (0.6, 0.6)), ("T4", (0.5, 0.5))])
def test_theta_star_returned_path_feasible_on_reported_heights(name, start):
    """Every returned segment obeys the grade cap at the z values it reports,
    and each reported z equals grid Z at that node.

    No minimum node count is asserted: Theta* may legitimately collapse to a
    single direct any-angle segment. The invariant is that WHATEVER path is
    returned is safe at its OWN heights -- which is exactly what the
    height-source fix guarantees (a pre-fix Theta* could return a 2-node direct
    shortcut whose snapped endpoints violate the grade cap).

    n=200 is the resolution at which the direct-segment grade violation on T4
    manifested before the direct-segment guard was added.
    """
    t = get_terrain(name)
    n = 200
    tlen, tpath = theta_star(t, n, start, t.sink)
    assert tpath is not None and len(tpath) >= 2
    axis, Z = build_height_grid(t, n)
    for (x0, y0, z0), (x1, y1, z1) in zip(tpath[:-1], tpath[1:]):
        horiz = math.hypot(x1 - x0, y1 - y0)
        if horiz > 0:
            assert abs(z1 - z0) / horiz <= MAX_GRADE_SLOPE + 1e-9, (
                f"{name}: returned Theta* segment grade "
                f"{abs(z1 - z0) / horiz:.4f} exceeds cap {MAX_GRADE_SLOPE:.4f}"
            )
    # reported z values must match grid Z (no analytic drift)
    for (x, y, z) in tpath:
        c = int(np.argmin(np.abs(axis - x)))
        r = int(np.argmin(np.abs(axis - y)))
        assert z == pytest.approx(float(Z[r, c]), rel=1e-12, abs=1e-12)
