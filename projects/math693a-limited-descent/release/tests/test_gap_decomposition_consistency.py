"""End-to-end consistency of the optimality-gap decomposition (plan core claim).

These tests wire the REAL reference pipeline (prepare_terrain -> Dijkstra ->
Theta* -> strategy walk) into the E5 gap formulas, catching pitfalls the
unit-level E5 tests (which use hand-built _Setup stubs) cannot:

  * Quantisation bias must be NON-NEGATIVE: QB = (raw - theta)/theta and Theta*
    smoothing can only shorten or preserve a path, so a negative QB would mean
    the "optimum" reference is internally inconsistent (theta longer than raw)
    -- which would silently corrupt every COG.
  * The run_e5 inline qb_pair formula must equal _gaps_for_trial's qb for the
    SAME (setup, start): a drift between the two definitions would break the
    aggregate identity COG = OG - QB that H1/H2 rest on.
  * COG = OG - QB must hold for real strategy trials on real terrain, not just
    for the algebraic stub.
  * Every start kept by prepare_terrain must have a positive, finite reference
    length (theta_len > 0) so the gap denominator is well defined -- a start
    that leaked through with theta_len == 0 would make OG undefined / inf.

We use small grids so the whole file runs in a few seconds. Terrains are keyed
by canonical T-codes (T1 == elliptic bowl control, T2 == Rosenbrock ridge).
"""
import math

import numpy as np
import pytest

from src.common import prepare_terrain
from src.strategies import run_strategy, DS_DEFAULT
from src.experiment_e5 import _gaps_for_trial

SMALL_N = 70


def _setups():
    # one smooth control + one non-trivial terrain so both regimes are exercised
    for name in ("T1", "T2"):
        yield name, prepare_terrain(name, SMALL_N, seed=0)


def test_reference_lengths_positive_and_finite():
    for name, setup in _setups():
        assert setup.starts, f"{name}: no usable starts"
        for sp in setup.starts:
            theta = setup.theta_len[sp]
            raw = setup.raw_len[sp]
            assert math.isfinite(theta) and theta > 0.0, (name, sp, theta)
            assert math.isfinite(raw) and raw > 0.0, (name, sp, raw)


def test_quantisation_bias_nonnegative():
    """Theta* smoothing never lengthens, so raw >= theta and QB >= 0.

    A negative QB here would mean the smoothed 'optimum' is LONGER than the raw
    Dijkstra path -- an internally inconsistent reference that would bias OG.
    """
    for name, setup in _setups():
        for sp in setup.starts:
            raw = setup.raw_len[sp]
            theta = setup.theta_len[sp]
            qb = (raw - theta) / theta
            assert qb >= -1e-9, f"{name} {sp}: negative QB {qb}"
            # the raw path must be at least as long as the smoothed optimum
            assert raw >= theta - 1e-9


def test_run_e5_qb_pair_matches_gaps_for_trial_qb():
    """The inline qb_pair formula used in run_e5 must equal _gaps_for_trial's
    qb for the same (setup, start). A divergence would silently break the
    aggregate COG = OG - QB bookkeeping (different QB per code path)."""
    for name, setup in _setups():
        for sp in setup.starts:
            # inline definition copied from run_e5
            inline_qb = (setup.raw_len[sp] - setup.theta_len[sp]) / setup.theta_len[sp] \
                if setup.theta_len[sp] > 0 else 0.0
            # use any real strategy trial to drive _gaps_for_trial
            res = run_strategy(setup.terrain, sp, "rotation_cw", ds=DS_DEFAULT, seed=0)
            gaps = _gaps_for_trial(setup, sp, res)
            assert gaps is not None
            _og, qb, _cog = gaps
            assert qb == pytest.approx(inline_qb, rel=1e-12, abs=1e-12)


def test_cog_equals_og_minus_qb_on_real_trials():
    """COG = OG - QB must hold for genuine strategy walks on real terrain."""
    checked = 0
    for name, setup in _setups():
        for sp in setup.starts:
            for strat in ("rotation_cw", "gradient_projection",
                          "unconstrained_steepest_descent"):
                res = run_strategy(setup.terrain, sp, strat,
                                   ds=DS_DEFAULT, seed=0)
                gaps = _gaps_for_trial(setup, sp, res)
                if gaps is None:
                    continue
                og, qb, cog = gaps
                assert cog == pytest.approx(og - qb, rel=1e-12, abs=1e-12)
                # OG must be defined relative to the SAME theta denominator
                expected_og = (res["path_length_3d"] - setup.theta_len[sp]) \
                    / setup.theta_len[sp]
                assert og == pytest.approx(expected_og, rel=1e-12, abs=1e-12)
                checked += 1
    assert checked > 0, "no real trials exercised the decomposition"


def test_starts_unique_and_reachable_via_prepare():
    """prepare_terrain must only retain starts whose sink is reachable (it has
    a reconstructed reference path) and must not duplicate a start cell --
    duplicates would double-weight a (terrain,start) pair in the bootstrap."""
    for name, setup in _setups():
        assert len(setup.starts) == len(set(setup.starts)), f"{name}: dup starts"
        for sp in setup.starts:
            assert sp in setup.raw_len and sp in setup.theta_len
            assert sp != setup.terrain.sink_ij
