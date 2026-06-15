"""Unit tests for autoscientist.verify.pitfalls."""

from __future__ import annotations

import pytest

from autoscientist.verify import pitfalls
from autoscientist.verify.pitfalls import (
    PitfallCheck,
    domain_config_path,
    load_pitfall_config,
    run_pitfalls,
)

# -- TOML loader -------------------------------------------------------------


def test_load_medical_imaging_toml():
    path = domain_config_path("medical_imaging")
    assert path.exists()
    checks = load_pitfall_config(path)
    ids = {c.id for c in checks}
    assert ids == {
        "patient_level_split",
        "site_stratification",
        "external_validation_present",
        "no_test_time_augmentation_in_baseline",
        "counterintuitive_signs_flagged",
        "baseline_reproduced_within_tolerance",
        "class_balance_reported",
        "multi_seed_reporting",
        "hyperparameter_tuning_split",
        "weak_label_provenance_disclosed",
        "view_projection_documented",
        "confidence_intervals_reported",
    }
    by_id = {c.id: c for c in checks}
    assert by_id["patient_level_split"].severity == "fail"
    assert by_id["no_test_time_augmentation_in_baseline"].severity == "needs_human"
    assert by_id["class_balance_reported"].severity == "warn"
    assert by_id["multi_seed_reporting"].severity == "fail"
    assert by_id["hyperparameter_tuning_split"].severity == "fail"
    assert by_id["weak_label_provenance_disclosed"].severity == "needs_human"
    assert by_id["view_projection_documented"].severity == "warn"
    assert by_id["confidence_intervals_reported"].severity == "needs_human"


def test_every_medical_imaging_check_has_a_handler():
    """Phase 7 invariant: registry covers every TOML id (no silent skips)."""
    checks = load_pitfall_config(domain_config_path("medical_imaging"))
    missing = [c.id for c in checks if pitfalls.get_handler(c.id) is None]
    assert missing == [], f"unhandled pitfall ids: {missing}"


def test_load_pitfall_config_rejects_duplicates(tmp_path):
    p = tmp_path / "dup.toml"
    p.write_text(
        '[[checks]]\nid = "a"\ntitle = "A"\nseverity = "fail"\n'
        '[[checks]]\nid = "a"\ntitle = "A again"\nseverity = "fail"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate pitfall id"):
        load_pitfall_config(p)


def test_load_pitfall_config_rejects_invalid_severity(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text(
        '[[checks]]\nid = "x"\ntitle = "X"\nseverity = "explode"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid severity"):
        load_pitfall_config(p)


# -- handler dispatch --------------------------------------------------------


def _by_id(verdicts, cid):
    return next(v for v in verdicts if v.check_id == cid)


def test_patient_level_split_pass_and_fail():
    state_pass = {"split_strategy": "patient"}
    state_fail = {"split_strategy": "image_level"}
    state_skip: dict = {}
    pf_pass = run_pitfalls(state_pass, domain="medical_imaging")
    pf_fail = run_pitfalls(state_fail, domain="medical_imaging")
    pf_skip = run_pitfalls(state_skip, domain="medical_imaging")
    assert _by_id(pf_pass, "patient_level_split").status == "pass"
    assert _by_id(pf_fail, "patient_level_split").status == "fail"
    assert _by_id(pf_skip, "patient_level_split").status == "skipped"


def test_site_stratification_logic():
    # Single source → pass automatically
    out = run_pitfalls(
        {"multi_source": False, "sites": ["NIH"]},
        domain="medical_imaging",
    )
    assert _by_id(out, "site_stratification").status == "pass"

    # Multi-source with stratification → pass
    out = run_pitfalls({
        "multi_source": True, "sites": ["NIH", "PadChest"],
        "site_stratified": True,
    }, domain="medical_imaging")
    assert _by_id(out, "site_stratification").status == "pass"

    # Multi-source, no stratification, no per-site → fail
    out = run_pitfalls({
        "multi_source": True, "sites": ["NIH", "PadChest"],
        "site_stratified": False,
    }, domain="medical_imaging")
    assert _by_id(out, "site_stratification").status == "fail"

    # Multi-source, per-site reporting alone is sufficient
    out = run_pitfalls({
        "multi_source": True, "sites": ["NIH", "PadChest"],
        "per_site_metrics": True,
    }, domain="medical_imaging")
    assert _by_id(out, "site_stratification").status == "pass"


def test_external_validation_logic():
    # Don't claim generalization → pass
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "external_validation_present").status == "pass"
    # Claim with external dataset → pass
    out = run_pitfalls({
        "claims_generalization": True,
        "external_validation_datasets": ["PadChest"],
    }, domain="medical_imaging")
    assert _by_id(out, "external_validation_present").status == "pass"
    # Claim without external dataset → fail
    out = run_pitfalls({"claims_generalization": True}, domain="medical_imaging")
    assert _by_id(out, "external_validation_present").status == "fail"


def test_tta_logic():
    # No flag → skipped (operator must declare)
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "no_test_time_augmentation_in_baseline").status == "skipped"
    # No TTA → pass
    out = run_pitfalls(
        {"test_time_augmentation": False}, domain="medical_imaging",
    )
    assert _by_id(out, "no_test_time_augmentation_in_baseline").status == "pass"
    # TTA + parity → pass
    out = run_pitfalls({
        "test_time_augmentation": True, "tta_baseline_match": True,
    }, domain="medical_imaging")
    assert _by_id(out, "no_test_time_augmentation_in_baseline").status == "pass"
    # TTA without parity → needs_human
    out = run_pitfalls({
        "test_time_augmentation": True, "tta_baseline_match": False,
    }, domain="medical_imaging")
    assert _by_id(out, "no_test_time_augmentation_in_baseline").status == "needs_human"


def test_counterintuitive_signs_explicit_and_inferred():
    # Explicit sign match → pass
    out = run_pitfalls({
        "hypothesized_effects": {"x": "positive"},
        "observed_effects": {"x": {"sign": "positive"}},
    }, domain="medical_imaging")
    assert _by_id(out, "counterintuitive_signs_flagged").status == "pass"

    # Inversion → fail
    out = run_pitfalls({
        "hypothesized_effects": {"x": "positive"},
        "observed_effects": {"x": {"sign": "negative"}},
    }, domain="medical_imaging")
    assert _by_id(out, "counterintuitive_signs_flagged").status == "fail"

    # Sign inferred from value
    out = run_pitfalls({
        "hypothesized_effects": {"x": "positive"},
        "observed_effects": {"x": {"value": -0.3}},
    }, domain="medical_imaging")
    assert _by_id(out, "counterintuitive_signs_flagged").status == "fail"

    # Aliases (+/- and up/down)
    out = run_pitfalls({
        "hypothesized_effects": {"x": "+", "y": "down"},
        "observed_effects": {
            "x": {"sign": "+"},
            "y": {"sign": "negative"},
        },
    }, domain="medical_imaging")
    assert _by_id(out, "counterintuitive_signs_flagged").status == "pass"


def test_baseline_pitfall_inherits_aggregate_status():
    out = run_pitfalls({
        "baseline_claims": [
            {"name": "a", "dataset": "d", "metric": "m",
             "published_value": 0.5, "observed_value": 0.51,
             "tolerance_abs": 0.02},
        ],
    }, domain="medical_imaging")
    v = _by_id(out, "baseline_reproduced_within_tolerance")
    assert v.status == "pass"
    assert v.category == "pitfall"

    out = run_pitfalls({"claims_novelty": True}, domain="medical_imaging")
    v = _by_id(out, "baseline_reproduced_within_tolerance")
    assert v.status == "fail"


def test_class_balance_reported_logic():
    out = run_pitfalls({
        "class_counts_train": {"a": 10, "b": 90},
        "class_counts_test": {"a": 5, "b": 45},
    }, domain="medical_imaging")
    assert _by_id(out, "class_balance_reported").status == "pass"

    out = run_pitfalls({
        "class_counts_train": {"a": 10, "b": 90},
    }, domain="medical_imaging")
    assert _by_id(out, "class_balance_reported").status == "fail"


def test_handler_crash_becomes_error_verdict(tmp_path):
    """A handler that raises must produce an ``error`` verdict, not propagate."""
    fixture = tmp_path / "crash.toml"
    fixture.write_text(
        '[[checks]]\nid = "crash_check"\ntitle = "Crash"\nseverity = "fail"\n',
        encoding="utf-8",
    )

    def boom(state, check):
        raise RuntimeError("intentional")

    pitfalls.register_check("crash_check", boom)
    try:
        out = run_pitfalls({}, domain="x", config_path=fixture)
        v = _by_id(out, "crash_check")
        assert v.status == "error"
        assert "intentional" in v.detail
    finally:
        pitfalls._REGISTRY.pop("crash_check", None)


def test_unregistered_check_id_skipped(tmp_path):
    fixture = tmp_path / "novel.toml"
    fixture.write_text(
        '[[checks]]\nid = "novel_check"\ntitle = "Novel"\nseverity = "fail"\n',
        encoding="utf-8",
    )
    out = run_pitfalls({}, domain="x", config_path=fixture)
    assert out[0].check_id == "novel_check"
    assert out[0].status == "skipped"


def test_run_pitfalls_missing_config_returns_empty(tmp_path):
    out = run_pitfalls({}, domain="x", config_path=tmp_path / "absent.toml")
    assert out == []


def test_pitfall_check_dataclass_basic():
    pc = PitfallCheck(id="x", title="X", severity="fail", description="d")
    assert pc.severity == "fail"
    assert pc.id == "x"


# -- Phase 7 additions -------------------------------------------------------


def test_multi_seed_reporting_logic():
    # Skipped when neither key present
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "multi_seed_reporting").status == "skipped"

    # < 3 seeds (declared via list) -> fail
    out = run_pitfalls({"seeds": [0, 1]}, domain="medical_imaging")
    v = _by_id(out, "multi_seed_reporting")
    assert v.status == "fail"
    assert "2" in v.detail

    # 3 seeds with explicit variance flag -> pass
    out = run_pitfalls(
        {"seeds": [0, 1, 2], "report_seed_variance": True},
        domain="medical_imaging",
    )
    assert _by_id(out, "multi_seed_reporting").status == "pass"

    # >= 3 seeds, variance inferred from per_condition_variance -> pass
    out = run_pitfalls(
        {"n_seeds": 5, "per_condition_variance": {"N=1k": 0.012}},
        domain="medical_imaging",
    )
    assert _by_id(out, "multi_seed_reporting").status == "pass"

    # >= 3 seeds, no variance evidence -> fail
    out = run_pitfalls({"n_seeds": 3}, domain="medical_imaging")
    v = _by_id(out, "multi_seed_reporting")
    assert v.status == "fail"
    assert "variance" in v.detail


def test_hyperparameter_tuning_split_logic():
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "hyperparameter_tuning_split").status == "skipped"

    for ok in ("validation", "val", "Dev", "cross_validation", "none"):
        out = run_pitfalls(
            {"hyperparameter_tuning_split": ok}, domain="medical_imaging",
        )
        assert _by_id(out, "hyperparameter_tuning_split").status == "pass", ok

    for bad in ("test", "holdout", "TRAIN", "held_out"):
        out = run_pitfalls(
            {"hyperparameter_tuning_split": bad}, domain="medical_imaging",
        )
        assert _by_id(out, "hyperparameter_tuning_split").status == "fail", bad

    out = run_pitfalls(
        {"hyperparameter_tuning_split": "kfold-magic"},
        domain="medical_imaging",
    )
    assert _by_id(out, "hyperparameter_tuning_split").status == "needs_human"


def test_weak_label_provenance_logic():
    # Skipped when not declared
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "weak_label_provenance_disclosed").status == "skipped"

    # All strong -> pass
    out = run_pitfalls({
        "label_provenance": {"InternalCohort": "expert_radiologist"},
    }, domain="medical_imaging")
    assert _by_id(out, "weak_label_provenance_disclosed").status == "pass"

    # Weak labels, undisclosed -> needs_human
    out = run_pitfalls({
        "label_provenance": {"NIH": "nlp_derived", "PadChest": "nlp_derived"},
    }, domain="medical_imaging")
    v = _by_id(out, "weak_label_provenance_disclosed")
    assert v.status == "needs_human"
    assert "NIH" in v.evidence["weak_datasets"]
    assert "PadChest" in v.evidence["weak_datasets"]

    # Weak labels, disclosed -> pass
    out = run_pitfalls({
        "label_provenance": {"NIH": "nlp_derived"},
        "weak_label_limitation_disclosed": True,
    }, domain="medical_imaging")
    assert _by_id(out, "weak_label_provenance_disclosed").status == "pass"

    # Unknown provenance string -> needs_human (don't silently pass)
    out = run_pitfalls({
        "label_provenance": {"Mystery": "vibes"},
    }, domain="medical_imaging")
    v = _by_id(out, "weak_label_provenance_disclosed")
    assert v.status == "needs_human"
    assert v.evidence["unknown_provenance"]


def test_view_projection_documented_logic():
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "view_projection_documented").status == "skipped"

    # Single view -> pass
    out = run_pitfalls(
        {"view_projections": ["PA"]}, domain="medical_imaging",
    )
    assert _by_id(out, "view_projection_documented").status == "pass"

    # Mixed views, no handling -> fail
    out = run_pitfalls(
        {"view_projections": ["PA", "AP"]}, domain="medical_imaging",
    )
    v = _by_id(out, "view_projection_documented")
    assert v.status == "fail"
    assert v.severity == "warn"  # warn-severity does not block

    # Mixed views, filtered -> pass
    out = run_pitfalls({
        "view_projections": ["PA", "AP"],
        "view_projection_filtered": True,
    }, domain="medical_imaging")
    assert _by_id(out, "view_projection_documented").status == "pass"

    # Mixed views, per-view metrics -> pass
    out = run_pitfalls({
        "view_projections": ["PA", "AP", "lateral"],
        "per_view_metrics": True,
    }, domain="medical_imaging")
    assert _by_id(out, "view_projection_documented").status == "pass"

    # Wrong type -> fail
    out = run_pitfalls(
        {"view_projections": "PA"}, domain="medical_imaging",
    )
    assert _by_id(out, "view_projection_documented").status == "fail"


def test_confidence_intervals_reported_logic():
    out = run_pitfalls({}, domain="medical_imaging")
    assert _by_id(out, "confidence_intervals_reported").status == "skipped"

    # No comparison claims -> pass (CIs unnecessary)
    out = run_pitfalls(
        {"comparison_claims": []}, domain="medical_imaging",
    )
    assert _by_id(out, "confidence_intervals_reported").status == "pass"

    # Comparison without CIs -> needs_human
    out = run_pitfalls({
        "comparison_claims": ["N=100k beats N=1k on PadChest"],
    }, domain="medical_imaging")
    v = _by_id(out, "confidence_intervals_reported")
    assert v.status == "needs_human"

    # Comparison with CIs -> pass
    out = run_pitfalls({
        "comparison_claims": ["N=100k beats N=1k"],
        "confidence_intervals_present": True,
        "confidence_interval_method": "bootstrap_2000",
    }, domain="medical_imaging")
    v = _by_id(out, "confidence_intervals_reported")
    assert v.status == "pass"
    assert "bootstrap_2000" in v.detail
