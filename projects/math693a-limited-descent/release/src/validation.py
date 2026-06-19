"""E0 validation checks: unique-minimum, gradient correctness, start sampling.

Each function returns a structured dict so the E0 driver can serialize results
and apply pass/fail thresholds from src.config.
"""
import numpy as np
from scipy import ndimage

from src.config import (
    DOMAIN_MIN, DOMAIN_MAX, GRID_UNIQUEMIN, UNIQUE_MIN_DEPTH_MARGIN,
    N_GRAD_CHECK_POINTS, GRAD_FD_TOL, START_MIN, START_MAX,
)


def verify_unique_minimum(terrain, n=GRID_UNIQUEMIN):
    """Verify the global min beats the next-lowest local min by >= margin.

    Returns dict with global_min, second_local_min, depth_margin, passed.
    Local minima are found on the discretized grid via a 3x3 minimum filter.
    """
    axis = np.linspace(DOMAIN_MIN, DOMAIN_MAX, n)
    gx, gy = np.meshgrid(axis, axis)
    Z = np.asarray(terrain.height(gx, gy), dtype=float)

    # Local minima: equal to local min filter and strictly below neighbors.
    local_min_filt = ndimage.minimum_filter(Z, size=3, mode="nearest")
    is_local_min = (Z == local_min_filt)

    minvals = Z[is_local_min]
    minvals = np.sort(minvals)
    global_min = float(minvals[0])

    # Find second-lowest *distinct* local minimum value (not adjacent plateau).
    # Use label-based grouping of the local-min mask to count separate basins.
    labeled, num = ndimage.label(is_local_min)
    basin_vals = []
    for lab in range(1, num + 1):
        basin_vals.append(float(Z[labeled == lab].min()))
    basin_vals = np.sort(np.asarray(basin_vals))

    if len(basin_vals) >= 2:
        second = float(basin_vals[1])
    else:
        second = float("inf")  # only one basin -> trivially unique

    depth_margin = second - global_min
    passed = bool(depth_margin >= UNIQUE_MIN_DEPTH_MARGIN)

    # Location of global minimum
    gidx = np.argmin(Z)
    gmin_loc = (float(gx.flat[gidx]), float(gy.flat[gidx]))

    return {
        "terrain": terrain.name,
        "global_min": global_min,
        "global_min_loc": gmin_loc,
        "n_basins": int(num),
        "second_local_min": second,
        "depth_margin": depth_margin,
        "required_margin": UNIQUE_MIN_DEPTH_MARGIN,
        "passed": passed,
    }


def verify_gradient(terrain, n_points=N_GRAD_CHECK_POINTS, eps=1e-6, rng=None):
    """Compare analytic gradient to central finite differences on random pts.

    Returns dict with max_abs_error and passed (<= GRAD_FD_TOL).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pts = rng.uniform(START_MIN, START_MAX, size=(n_points, 2))
    max_err = 0.0
    for x, y in pts:
        ax, ay = terrain.grad(x, y)
        ax = float(ax)
        ay = float(ay)
        fx = (float(terrain.height(x + eps, y)) -
              float(terrain.height(x - eps, y))) / (2 * eps)
        fy = (float(terrain.height(x, y + eps)) -
              float(terrain.height(x, y - eps))) / (2 * eps)
        max_err = max(max_err, abs(ax - fx), abs(ay - fy))
    return {
        "terrain": terrain.name,
        "max_abs_error": float(max_err),
        "tolerance": GRAD_FD_TOL,
        "passed": bool(max_err <= GRAD_FD_TOL),
    }


def sample_start_points(rng, n_points):
    """Latin-hypercube-style start points in [START_MIN, START_MAX]^2.

    Returns array of shape (n_points, 2). Uses stratified 1-D LHS per axis.
    """
    cuts = np.linspace(0.0, 1.0, n_points + 1)
    u = rng.uniform(size=(n_points, 2))
    samples = np.empty((n_points, 2))
    for d in range(2):
        strat = cuts[:-1][:, None].ravel() + u[:, d] * (1.0 / n_points)
        rng.shuffle(strat)
        samples[:, d] = START_MIN + strat * (START_MAX - START_MIN)
    return samples
