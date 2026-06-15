"""Unit tests for autoscientist.meta.ab_harness aggregation logic.

End-to-end harness behavior is covered by smoke_phase6.py — those tests
exercise the LLM call path. Here we only test the deterministic pieces:
variance, winner picking, ABResult shape.
"""

from __future__ import annotations

from autoscientist.meta.ab_harness import (
    ABResult,
    VariantResult,
    _pick_winner,
    _variance,
)


def _vr(idx, mean, var=0.0, parse=0):
    return VariantResult(
        variant_index=idx, prompt_version_id=f"pv_{idx}",
        hypothesis="h", per_anchor_totals={},
        mean_score=mean, score_variance=var, n_parse_errors=parse,
    )


def test_variance_zero_for_constant_or_singleton():
    assert _variance([4.0, 4.0, 4.0]) == 0.0
    assert _variance([4.0]) == 0.0
    assert _variance([]) == 0.0


def test_variance_known_value():
    # Population variance of [1, 3] = 1.0
    assert _variance([1.0, 3.0]) == 1.0


def test_winner_picks_highest_mean():
    idx = _pick_winner([_vr(0, 3.0), _vr(1, 4.5), _vr(2, 2.0)])
    assert idx == 1


def test_winner_breaks_tie_by_lower_variance():
    idx = _pick_winner([
        _vr(0, 4.0, var=2.0),
        _vr(1, 4.0, var=0.5),
        _vr(2, 4.0, var=1.0),
    ])
    assert idx == 1


def test_winner_breaks_double_tie_by_parse_errors():
    idx = _pick_winner([
        _vr(0, 4.0, var=1.0, parse=3),
        _vr(1, 4.0, var=1.0, parse=0),
        _vr(2, 4.0, var=1.0, parse=2),
    ])
    assert idx == 1


def test_winner_index_negative_for_empty():
    assert _pick_winner([]) == -1


def test_ab_result_winner_property():
    vs = (_vr(0, 3.0), _vr(1, 4.5))
    r = ABResult(agent="x", rubric="x", anchor_count=2, variants=vs, winner_index=1)
    assert r.winner is vs[1]
    r_none = ABResult(agent="x", rubric="x", anchor_count=2, variants=vs, winner_index=-1)
    assert r_none.winner is None


def test_ab_result_to_dict_round_trip():
    vs = (_vr(0, 3.0),)
    r = ABResult(agent="x", rubric="x", anchor_count=1, variants=vs, winner_index=0)
    d = r.to_dict()
    assert d["agent"] == "x"
    assert d["winner_index"] == 0
    assert isinstance(d["variants"], list) and len(d["variants"]) == 1
