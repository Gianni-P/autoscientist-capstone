"""Continuous descent strategies on the analytic terrain (plan E2/E3/E4).

Each strategy walks the *normalised* terrain surface from a continuous start
position (x, y) toward the sink (x, y), taking fixed planar step length ds.
At each step a candidate planar direction (unit vector in the (x,y) plane) is
chosen and the walker moves ds along it; the 3-D step length accounts for the
height change dz between the current and next surface point.

We evaluate the surface and its gradient by bilinear interpolation over the
precomputed normalised grid `terrain.z` (z[i,j] = f(xs[j], ys[i])), so the
strategies share the *exact same* surface the grid Dijkstra reference uses.
The gradient is obtained by finite differences of that interpolant
(plan: gradient_validation = finite_difference), matching the project default.

Grade of a planar step of horizontal length ds with height change dz is
|dz| / ds. A step is "feasible" if that grade <= MAX_GRADE_TAN.

Strategies
----------
unconstrained_steepest_descent  (E2) -- always move along -grad, no grade check.
rotation_heuristic              (E3) -- start from steepest-descent direction;
                                        if the step is infeasible, rotate the
                                        direction by +/-1 deg (CW or CCW) until a
                                        feasible step is found, up to 360 deg.
gradient_projection             (E4) -- if steepest-descent step is feasible take
                                        it; else pick the feasible-cone-boundary
                                        direction (nearest CW vs nearest CCW) with
                                        greater descent (lower next height).

Final snap-to-sink semantics
----------------------------
When the walker enters the convergence radius (2*ds) it takes one forced
terminal segment straight to the sink so the recorded length terminates exactly
at the optimum's endpoint (endpoints comparable, per the objective). This final
snap is a *forced terminal step*: its grade is recorded in ``max_grade`` and
surfaced separately as ``final_snap_grade`` / ``final_snap_feasible`` for
transparency. The snap IS grade-checked and **counted in ``n_violations`` when
it is infeasible** -- a path whose terminal segment exceeds MAX_GRADE_TAN is a
genuinely infeasible path and must not report ``n_violations`` = 0 /
``feasibility_rate`` = 1.0. On smooth terrains with a small ds the snap is
gentle, so it remains feasible and does not inflate ``n_violations``.
``final_snap_grade`` / ``final_snap_feasible`` remain as additional transparency
fields. Downstream feasibility of the heuristic *trajectory* reads
``n_violations`` (all infeasible steps, chosen + terminal snap) and
``feasibility_rate``, and may additionally inspect ``final_snap_feasible`` to
attribute an infeasibility specifically to the terminal segment.

Public API
----------
DS_DEFAULT, MAX_ITERS, ROT_INCREMENT_DEG
sink_xy(terrain) -> (x, y)
ij_to_xy(terrain, i, j) -> (x, y)
run_strategy(terrain, start_ij, strategy, ds, max_iters, seed) -> dict
STRATEGIES -> dict name->callable spec
"""
import math

from src.config import MAX_GRADE_TAN, DOMAIN_MIN, DOMAIN_MAX

DS_DEFAULT = 0.002
MAX_ITERS = 10000
ROT_INCREMENT_DEG = 1.0


def ij_to_xy(terrain, i, j):
    return float(terrain.xs[j]), float(terrain.ys[i])


def sink_xy(terrain):
    return float(terrain.sink_xy[0]), float(terrain.sink_xy[1])


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _bilinear_height(terrain, x, y):
    """Bilinearly interpolate the normalised height z at continuous (x, y)."""
    n = terrain.n
    x = _clamp(x, DOMAIN_MIN, DOMAIN_MAX)
    y = _clamp(y, DOMAIN_MIN, DOMAIN_MAX)
    # fractional grid coordinates: column index from x, row index from y
    fc = (x - DOMAIN_MIN) / terrain.dx if terrain.dx > 0 else 0.0
    fr = (y - DOMAIN_MIN) / terrain.dy if terrain.dy > 0 else 0.0
    j0 = int(math.floor(fc))
    i0 = int(math.floor(fr))
    j0 = _clamp(j0, 0, n - 2) if n > 1 else 0
    i0 = _clamp(i0, 0, n - 2) if n > 1 else 0
    j1 = min(j0 + 1, n - 1)
    i1 = min(i0 + 1, n - 1)
    tx = fc - j0
    ty = fr - i0
    tx = _clamp(tx, 0.0, 1.0)
    ty = _clamp(ty, 0.0, 1.0)
    z = terrain.z
    z00 = z[i0, j0]
    z01 = z[i0, j1]
    z10 = z[i1, j0]
    z11 = z[i1, j1]
    top = z00 * (1 - tx) + z01 * tx
    bot = z10 * (1 - tx) + z11 * tx
    return float(top * (1 - ty) + bot * ty)


def _fd_partial(terrain, x, y, axis, delta):
    """Finite-difference partial derivative along axis (0=x, 1=y).

    Uses a central difference when both probe points stay inside the domain,
    and falls back to a one-sided (forward/backward) difference with the
    correct denominator at the domain boundary, so the gradient *magnitude*
    is not artificially halved at the edges (see review minor on _fd_gradient).
    """
    if axis == 0:
        lo_ok = (x - delta) >= DOMAIN_MIN
        hi_ok = (x + delta) <= DOMAIN_MAX
        zp = _bilinear_height(terrain, x + delta, y)
        zm = _bilinear_height(terrain, x - delta, y)
        z0 = _bilinear_height(terrain, x, y)
    else:
        lo_ok = (y - delta) >= DOMAIN_MIN
        hi_ok = (y + delta) <= DOMAIN_MAX
        zp = _bilinear_height(terrain, x, y + delta)
        zm = _bilinear_height(terrain, x, y - delta)
        z0 = _bilinear_height(terrain, x, y)
    if lo_ok and hi_ok:
        return (zp - zm) / (2.0 * delta)
    if hi_ok:               # lo probe out of domain: use forward difference
        return (zp - z0) / delta
    if lo_ok:               # hi probe out of domain: use backward difference
        return (z0 - zm) / delta
    # domain narrower than delta in this axis: degenerate, no gradient info
    return 0.0


def _fd_gradient(terrain, x, y, delta=None):
    """Finite-difference gradient (dz/dx, dz/dy) of the interpolant.

    The default probe step is a *half* grid-cell spacing
    (0.5 * max(dx, dy)) -- a grid-resolution-scale finite difference. A full
    grid-cell delta over-smooths high-curvature surfaces at coarse (test) grids
    (review minor); half a cell keeps the magnitude closer to the local slope
    while still straddling at least one interpolation cell. Central difference
    in the interior; one-sided at domain boundaries so the magnitude stays
    correct (direction is unaffected either way). `delta` is exposed so callers
    can override the probe scale.
    """
    if delta is None:
        delta = 0.5 * max(terrain.dx, terrain.dy)
        if delta <= 0:
            delta = 1e-3
    gx = _fd_partial(terrain, x, y, 0, delta)
    gy = _fd_partial(terrain, x, y, 1, delta)
    return gx, gy


def _steepest_dir(terrain, x, y):
    """Unit planar direction of steepest descent (-grad / |grad|).

    Returns (ux, uy, grad_norm). If the gradient is ~0 (flat / at minimum)
    returns the direction toward the sink so the walker still progresses.
    """
    gx, gy = _fd_gradient(terrain, x, y)
    gnorm = math.hypot(gx, gy)
    if gnorm < 1e-12:
        sx, sy = sink_xy(terrain)
        dx, dy = sx - x, sy - y
        dnorm = math.hypot(dx, dy)
        if dnorm < 1e-12:
            return 1.0, 0.0, 0.0
        return dx / dnorm, dy / dnorm, 0.0
    return -gx / gnorm, -gy / gnorm, gnorm


def _rotate(ux, uy, deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return ux * c - uy * s, ux * s + uy * c


def _step_grade(terrain, x, y, ux, uy, ds):
    """Grade |dz|/ds and 3-D segment length for a step of length ds along (ux,uy)."""
    nx = x + ux * ds
    ny = y + uy * ds
    z0 = _bilinear_height(terrain, x, y)
    z1 = _bilinear_height(terrain, nx, ny)
    dz = z1 - z0
    grade = abs(dz) / ds if ds > 0 else float("inf")
    seg3d = math.sqrt(ds * ds + dz * dz)
    return grade, seg3d, nx, ny, z1


def _choose_direction(terrain, x, y, ds, strategy):
    """Return (ux, uy, grade, feasible, swept_deg) for the chosen step.

    strategy in {'unconstrained', 'rotation_cw', 'rotation_ccw', 'projection'}.
    swept_deg = angular sweep used (0 for unconstrained/feasible first try).
    feasible = whether the chosen step satisfies the grade constraint.
    """
    ux, uy, gnorm = _steepest_dir(terrain, x, y)
    grade, _seg, _nx, _ny, _z1 = _step_grade(terrain, x, y, ux, uy, ds)

    if strategy == "unconstrained":
        return ux, uy, grade, grade <= MAX_GRADE_TAN + 1e-12, 0.0

    # If steepest direction is already feasible, take it.
    if grade <= MAX_GRADE_TAN + 1e-12:
        return ux, uy, grade, True, 0.0

    if strategy in ("rotation_cw", "rotation_ccw"):
        sign = -1.0 if strategy == "rotation_cw" else 1.0
        deg = ROT_INCREMENT_DEG
        while deg <= 360.0 + 1e-9:
            rx, ry = _rotate(ux, uy, sign * deg)
            g, _s, _a, _b, _c = _step_grade(terrain, x, y, rx, ry, ds)
            if g <= MAX_GRADE_TAN + 1e-12:
                return rx, ry, g, True, deg
            deg += ROT_INCREMENT_DEG
        # no feasible direction found within 360 deg
        return ux, uy, grade, False, 360.0

    if strategy == "projection":
        # Find nearest feasible-cone-boundary direction on each side (CW, CCW)
        # by a fine angular search, then pick the candidate with greater descent
        # (lower next height). Boundary = smallest rotation reaching feasibility.
        best = None  # (next_height, rx, ry, g, deg)
        for sign in (-1.0, 1.0):
            deg = ROT_INCREMENT_DEG
            while deg <= 360.0 + 1e-9:
                rx, ry = _rotate(ux, uy, sign * deg)
                g, _s, _a, _b, z1 = _step_grade(terrain, x, y, rx, ry, ds)
                if g <= MAX_GRADE_TAN + 1e-12:
                    cand = (z1, rx, ry, g, deg)
                    if best is None or cand[0] < best[0]:
                        best = cand
                    break
                deg += ROT_INCREMENT_DEG
        if best is not None:
            _z, rx, ry, g, deg = best
            return rx, ry, g, True, deg
        return ux, uy, grade, False, 360.0

    raise ValueError(f"unknown strategy {strategy!r}")


# Map E3/E4 method names to internal strategy keys.
STRATEGIES = {
    "unconstrained_steepest_descent": "unconstrained",
    "rotation_cw": "rotation_cw",
    "rotation_ccw": "rotation_ccw",
    "gradient_projection": "projection",
}


def run_strategy(terrain, start_ij, strategy, ds=DS_DEFAULT,
                 max_iters=MAX_ITERS, seed=0):
    """Walk the terrain from start_ij toward the sink using `strategy`.

    Parameters
    ----------
    strategy : str  -- a key of STRATEGIES (public method name) OR an internal
        key ('unconstrained', 'rotation_cw', 'rotation_ccw', 'projection').
    seed : int -- retained for API/logging consistency; the walk is fully
        deterministic and does NOT touch any global RNG state.

    Returns a result dict with:
      converged (bool), iterations (int, == n_steps), path_length_3d (float),
      n_violations (int), n_steps (int), feasibility_rate (float),
      max_grade (float), final_distance_to_sink (float),
      final_snap_grade (float), final_snap_feasible (bool), reason (str).

    ``n_violations`` counts *every* step whose grade exceeds the constraint,
    including the forced terminal snap-to-sink when that snap is itself
    infeasible. The snap is grade-checked: if ``final_snap_grade`` exceeds
    MAX_GRADE_TAN the snap is counted in ``n_violations`` so a path with a steep
    terminal segment cannot report ``feasibility_rate`` = 1.0. Its grade is also
    reported separately via ``final_snap_grade`` / ``final_snap_feasible`` and
    folded into ``max_grade`` for transparency.
    """
    internal = STRATEGIES.get(strategy, strategy)
    valid = {"unconstrained", "rotation_cw", "rotation_ccw", "projection"}
    if internal not in valid:
        raise ValueError(f"unknown strategy {strategy!r}")

    x, y = ij_to_xy(terrain, start_ij[0], start_ij[1])
    sx, sy = sink_xy(terrain)
    converge_radius = 2.0 * ds

    total_len = 0.0
    n_violations = 0
    n_steps = 0
    max_grade = 0.0
    converged = False
    reason = "max_iters"
    final_snap_grade = 0.0
    final_snap_feasible = True

    constrained = internal != "unconstrained"

    for it in range(max_iters):
        dist_to_sink = math.hypot(sx - x, sy - y)
        if dist_to_sink <= converge_radius:
            # Forced terminal segment straight to the sink so the recorded
            # length terminates exactly at the optimum's endpoint. This is a
            # forced terminal step: its grade is folded into max_grade and
            # surfaced as final_snap_grade/final_snap_feasible. The snap IS
            # grade-checked and counted in n_violations when infeasible, so a
            # path whose terminal segment exceeds MAX_GRADE_TAN is not silently
            # reported as fully feasible.
            z_cur = _bilinear_height(terrain, x, y)
            z_sink = _bilinear_height(terrain, sx, sy)
            dz = z_sink - z_cur
            total_len += math.sqrt(dist_to_sink ** 2 + dz ** 2)
            if dist_to_sink > 1e-15:
                final_snap_grade = abs(dz) / dist_to_sink
                n_steps += 1
                max_grade = max(max_grade, final_snap_grade)
                final_snap_feasible = (
                    final_snap_grade <= MAX_GRADE_TAN + 1e-12)
                if final_snap_grade > MAX_GRADE_TAN + 1e-12:
                    n_violations += 1
            converged = True
            reason = "reached_sink"
            break

        ux, uy, grade, feasible, _swept = _choose_direction(
            terrain, x, y, ds, internal)

        if not feasible and constrained:
            # constrained strategy could not find a feasible direction
            reason = "no_feasible_direction"
            break

        _g, seg3d, nx, ny, _z1 = _step_grade(terrain, x, y, ux, uy, ds)
        total_len += seg3d
        n_steps += 1
        max_grade = max(max_grade, grade)
        if grade > MAX_GRADE_TAN + 1e-12:
            n_violations += 1

        # Detect being stuck (no spatial progress) -> treat as non-convergence.
        if abs(nx - x) < 1e-15 and abs(ny - y) < 1e-15:
            reason = "stuck"
            break
        x, y = nx, ny

    final_dist = math.hypot(sx - x, sy - y)
    # feasibility_rate measures the fraction of steps that respected the grade.
    # The forced terminal snap is counted in n_steps and, when infeasible, also
    # in n_violations -- so a steep terminal snap correctly drags the rate below
    # 1.0, while a benign (gentle) snap leaves an otherwise-feasible walk at 1.0.
    feas_rate = (1.0 - n_violations / n_steps) if n_steps > 0 else 1.0

    return {
        "strategy": strategy,
        "converged": bool(converged),
        # `iterations` is kept as an alias of `n_steps` (number of segments
        # actually taken, including the final snap-to-sink) for downstream
        # consumers (E3/E4/E5 records). They are intentionally equal.
        "iterations": int(n_steps),
        "path_length_3d": float(total_len),
        "n_violations": int(n_violations),
        "n_steps": int(n_steps),
        "feasibility_rate": float(feas_rate),
        "max_grade": float(max_grade),
        "final_distance_to_sink": float(final_dist),
        "final_snap_grade": float(final_snap_grade),
        "final_snap_feasible": bool(final_snap_feasible),
        "ds": float(ds),
        "reason": reason,
    }
