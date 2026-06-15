"""Unit tests for autoscientist.verify.leakage."""

from __future__ import annotations

from autoscientist.verify import leakage


def test_id_overlap_disjoint_passes():
    v = leakage.check_id_overlap(
        train_ids=["p1", "p2"], test_ids=["p3", "p4"], val_ids=["p5"],
    )
    assert v.status == "pass"
    assert v.evidence["n_train"] == 2
    assert v.evidence["n_test"] == 2
    assert v.evidence["n_val"] == 1


def test_id_overlap_train_test_fails():
    v = leakage.check_id_overlap(
        train_ids=["p1", "p2", "p3"], test_ids=["p3", "p4"],
    )
    assert v.status == "fail"
    assert v.evidence["overlapping_pairs"]["train_test"] == ["p3"]


def test_id_overlap_three_way_records_each_pair():
    v = leakage.check_id_overlap(
        train_ids=["a", "b"], test_ids=["b", "c"], val_ids=["a", "c"],
    )
    pairs = v.evidence["overlapping_pairs"]
    assert v.status == "fail"
    assert pairs["train_test"] == ["b"]
    assert pairs["train_val"] == ["a"]
    assert pairs["val_test"] == ["c"]


def test_id_overlap_missing_inputs_skipped():
    v = leakage.check_id_overlap(train_ids=None, test_ids=None)
    assert v.status == "skipped"


def test_id_overlap_handles_int_and_str_ids():
    v = leakage.check_id_overlap(
        train_ids=[1, 2, 3], test_ids=["3", 4],   # 3 vs "3" coerces to "3"
    )
    assert v.status == "fail"
    assert v.evidence["overlapping_pairs"]["train_test"] == ["3"]


# -- target leakage ----------------------------------------------------------


def test_target_leakage_perfect_correlation_fails():
    target = [0, 1, 0, 1, 0, 1, 0, 1]
    leaky = [0, 1, 0, 1, 0, 1, 0, 1]
    v = leakage.check_target_leakage(features={"f": leaky}, target=target)
    assert v.status == "fail"


def test_target_leakage_constant_feature_does_not_flag():
    """Regression: sort tiebreaker (value, target) used to make a constant
    feature fake perfect single-feature accuracy."""
    target = [0, 1, 0, 1, 0, 1, 0, 1]
    v = leakage.check_target_leakage(
        features={"const": [3.14] * 8}, target=target,
    )
    assert v.status == "pass"


def test_target_leakage_overlapping_classes_passes():
    target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    benign = [0.5, 0.5, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5]
    v = leakage.check_target_leakage(features={"f": benign}, target=target)
    assert v.status == "pass"


def test_target_leakage_skipped_without_inputs():
    v = leakage.check_target_leakage(features=None, target=None)
    assert v.status == "skipped"


def test_target_leakage_multiple_features_only_leaky_listed():
    target = [0, 1, 0, 1, 0, 1, 0, 1]
    v = leakage.check_target_leakage(
        features={
            "leaky": [0, 100, 0, 100, 0, 100, 0, 100],
            "noisy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        },
        target=target,
    )
    assert v.status == "fail"
    flagged = {row["feature"] for row in v.evidence["suspicious"]}
    assert "leaky" in flagged
    assert "noisy" not in flagged


def test_target_leakage_too_few_samples_skipped():
    v = leakage.check_target_leakage(
        features={"f": [0, 1, 0]}, target=[0, 1, 0],
    )
    assert v.status == "skipped"


def test_run_leakage_emits_two_verdicts():
    state = {
        "train_ids": ["a"], "test_ids": ["b"],
        "features": {"f": [0, 1, 0, 1, 0, 1, 0, 1]},
        "target": [0, 1, 0, 1, 0, 1, 0, 1],
    }
    out = leakage.run_leakage(state)
    assert {v.check_id for v in out} == {"id_overlap", "target_leakage"}
