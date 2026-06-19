"""Tests for start-point sampling correctness and seed determinism.

Pitfalls guarded:
  * Start points must stay inside the configured sampling box (no leaky
    out-of-domain starts that would invalidate the feasibility comparison).
  * Shape of the sampled array must be (n, 2) -- a silent shape mismatch would
    break the per-start loop in run_e0.
  * Determinism: the SAME seed must reproduce identical sampling and identical
    E0 summary; DIFFERENT seeds must actually change the sampled points.
"""
import numpy as np
import pytest

from src.config import START_MIN, START_MAX
from src.validation import sample_start_points


def test_sample_shape_and_bounds():
    rng = np.random.default_rng(0)
    n = 8
    pts = sample_start_points(rng, n)
    assert pts.shape == (n, 2)
    assert np.all(pts >= START_MIN - 1e-9)
    assert np.all(pts <= START_MAX + 1e-9)


def test_sampling_is_deterministic_for_same_seed():
    a = sample_start_points(np.random.default_rng(123), 8)
    b = sample_start_points(np.random.default_rng(123), 8)
    assert np.allclose(a, b)


def test_sampling_differs_for_different_seed():
    a = sample_start_points(np.random.default_rng(1), 8)
    b = sample_start_points(np.random.default_rng(2), 8)
    assert not np.allclose(a, b)


def test_sampling_covers_strata():
    # LHS should spread points: with 8 points per axis no single bin should hold
    # all of them. Check that the spread across the box is non-degenerate.
    pts = sample_start_points(np.random.default_rng(7), 8)
    spread_x = pts[:, 0].max() - pts[:, 0].min()
    spread_y = pts[:, 1].max() - pts[:, 1].min()
    box = START_MAX - START_MIN
    assert spread_x > 0.4 * box
    assert spread_y > 0.4 * box


def test_run_e0_summary_is_seed_deterministic():
    # End-to-end determinism on a tiny grid: same seed -> identical gate summary.
    from src.e0 import run_e0
    kw = dict(grid_primary=24, grid_validation=16, n_starts=2)
    s1 = run_e0(seed=0, run_id="pytest_det_a", **kw)
    s2 = run_e0(seed=0, run_id="pytest_det_b", **kw)
    # Numeric / boolean gate fields must match exactly across runs.
    for key in ["all_pass", "n_path_pairs", "n_feasible",
                "all_theta_le_dijkstra", "t1_baseline_len"]:
        assert s1[key] == s2[key], f"non-deterministic field: {key}"
