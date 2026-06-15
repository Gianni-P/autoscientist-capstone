"""Phase 5 smoke test — verification harness.

KICKOFF.md §9 Phase 5 — verify/* exists and produces deterministic
Pass / Fail / Needs-human verdicts. This smoke exercises:

  * Pure-library correctness across every verify module on synthetic
    pipeline-state dicts that hit each verdict class.
  * The orchestrator (:func:`verify.run_all`) — outcome aggregation
    across leakage, baseline_repro, stats, and pitfalls; correct
    deduplication of the baseline-aggregate verdict.
  * The checkpoint integration: a non-clean report opens a stage-4
    checkpoint with the report serialized into ``parsed``; a clean
    report opens nothing.

Zero LLM spend — verify is fully deterministic and offline.

    uv run python scripts/smoke_phase5.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_DB = _REPO / "smoke_phase5.db"
_RUNS = _REPO / "runs_smoke_phase5"
if _DB.exists():
    _DB.unlink()
if _RUNS.exists():
    shutil.rmtree(_RUNS)

os.environ["AUTOSCIENTIST_DB_PATH"] = str(_DB)


def section(name: str) -> None:
    print(f"\n=== {name} ===")


def passed(msg: str) -> None:
    print(f"  PASS  {msg}")


def fail(msg: str) -> None:
    raise AssertionError(msg)


def expect(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)
    passed(msg)


def main() -> int:
    from autoscientist.checkpoints import manager as cp_manager
    from autoscientist.state.db import open_db, start_run
    from autoscientist.verify import (
        baseline_repro,
        leakage,
        pitfalls,
        run_all,
        stats,
    )
    from autoscientist.verify.pitfalls import (
        domain_config_path,
        load_pitfall_config,
    )

    # ------------------------------------------------------------------
    # 1) Pitfall TOML loads cleanly with the expected ids.
    # ------------------------------------------------------------------
    section("Pitfall config: medical_imaging.toml loads with expected ids")
    cfg_path = domain_config_path("medical_imaging")
    expect(cfg_path.exists(), f"medical_imaging.toml exists at {cfg_path}")
    checks = load_pitfall_config(cfg_path)
    ids = {c.id for c in checks}
    required = {
        "patient_level_split",
        "site_stratification",
        "external_validation_present",
        "no_test_time_augmentation_in_baseline",
        "counterintuitive_signs_flagged",
        "baseline_reproduced_within_tolerance",
        "class_balance_reported",
    }
    expect(required <= ids,
           f"all 7 medical-imaging pitfall ids present "
           f"(missing={required - ids})")
    sevs = {c.id: c.severity for c in checks}
    expect(sevs["patient_level_split"] == "fail",
           "patient_level_split has severity 'fail'")
    expect(sevs["no_test_time_augmentation_in_baseline"] == "needs_human",
           "no_test_time_augmentation_in_baseline has severity 'needs_human'")
    expect(sevs["class_balance_reported"] == "warn",
           "class_balance_reported has severity 'warn'")

    # ------------------------------------------------------------------
    # 2) Leakage detector — id overlap and target leakage edge cases.
    # ------------------------------------------------------------------
    section("Leakage: id overlap detected, clean splits pass")
    v_overlap = leakage.check_id_overlap(
        train_ids=["p1", "p2", "p3", "p4"],
        test_ids=["p4", "p5"],
        val_ids=["p2"],
    )
    expect(v_overlap.status == "fail", "overlap → fail")
    expect("train_test" in v_overlap.evidence["overlapping_pairs"],
           "train_test pair recorded")
    expect("train_val" in v_overlap.evidence["overlapping_pairs"],
           "train_val pair recorded")

    v_clean = leakage.check_id_overlap(
        train_ids=["p1", "p2"], test_ids=["p3", "p4"], val_ids=["p5"],
    )
    expect(v_clean.status == "pass", "disjoint splits → pass")

    section("Leakage: target leakage on near-deterministic feature")
    target = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    leaky = [-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]   # perfect classifier
    # benign overlaps the classes — neither pearson high nor any
    # threshold separates 0s from 1s.
    benign = [0.5, 0.5, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5]
    v_leak = leakage.check_target_leakage(
        features={"leaky_col": leaky, "benign_col": benign},
        target=target,
    )
    expect(v_leak.status == "fail", "leaky feature → fail")
    suspicious_names = {row["feature"] for row in v_leak.evidence["suspicious"]}
    expect("leaky_col" in suspicious_names, "leaky_col flagged")
    expect("benign_col" not in suspicious_names, "benign_col not flagged")

    # Constant feature must NOT spuriously flag (regression test for the
    # sort-tiebreaker false positive).
    v_const = leakage.check_target_leakage(
        features={"const": [0.0] * 10},
        target=target,
    )
    expect(v_const.status == "pass",
           "constant feature does not trigger spurious target-leakage hit")

    # ------------------------------------------------------------------
    # 3) Baseline repro — within-tolerance, out-of-tolerance, novelty rule.
    # ------------------------------------------------------------------
    section("Baseline repro: tolerance behavior + novelty hard rule")
    state_in_tol = {
        "baseline_claims": [
            {
                "name": "CheXNet-Rajpurkar2017",
                "dataset": "NIH-ChestX-ray14",
                "metric": "AUROC-pneumonia",
                "published_value": 0.7680,
                "observed_value": 0.7710,
                "tolerance_abs": 0.01,
            },
        ],
    }
    verdicts = baseline_repro.run_baseline_repro(state_in_tol)
    aggregate = next(
        v for v in verdicts if v.check_id == "baseline_reproduced_within_tolerance"
    )
    expect(aggregate.status == "pass", "0.768→0.771 within ±0.01 → aggregate pass")

    state_out_tol = {
        "baseline_claims": [
            {
                "name": "CheXNet",
                "dataset": "NIH",
                "metric": "AUROC",
                "published_value": 0.768,
                "observed_value": 0.620,
                "tolerance_abs": 0.02,
            },
        ],
    }
    verdicts = baseline_repro.run_baseline_repro(state_out_tol)
    aggregate = next(
        v for v in verdicts if v.check_id == "baseline_reproduced_within_tolerance"
    )
    expect(aggregate.status == "fail",
           "0.620 vs 0.768 (tol 0.02) → aggregate fail")

    state_novelty_no_baseline = {"claims_novelty": True}
    verdicts = baseline_repro.run_baseline_repro(state_novelty_no_baseline)
    aggregate = next(
        v for v in verdicts if v.check_id == "baseline_reproduced_within_tolerance"
    )
    expect(aggregate.status == "fail",
           "novelty claimed without any baseline → fail (KICKOFF §4 #7)")
    expect(aggregate.evidence.get("novelty_claimed") is True,
           "novelty rule flagged in evidence")

    # ------------------------------------------------------------------
    # 4) Stats — multicollinearity, normality, sample size.
    # ------------------------------------------------------------------
    section("Stats: multicollinearity, normality, sample size")
    v_mc = stats.check_multicollinearity({
        "x": [1, 2, 3, 4, 5, 6, 7, 8],
        "y": [2, 4, 6, 8, 10, 12, 14, 16],   # perfect linear of x
        "z": [9, 1, 4, 2, 7, 3, 5, 8],
    })
    expect(v_mc.status == "fail", "perfect collinearity → fail")
    expect(any("x" in pair["a"] or "x" in pair["b"]
               for pair in v_mc.evidence["fails"]),
           "x is named in the fail evidence")

    v_norm_skewed = stats.check_normality({
        # one big outlier → skewness ≈ 2.67, comfortably past the 2.0 rule
        "residuals": [1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
    })
    expect(v_norm_skewed.status == "needs_human",
           "heavy-tailed distribution → needs_human")

    # Reasonably symmetric small sample → pass.
    v_norm_ok = stats.check_normality({
        "x": [-2, -1, -1, 0, 0, 0, 0, 1, 1, 2],
    })
    expect(v_norm_ok.status == "pass", "balanced distribution → pass")

    v_size_fail = stats.check_sample_size(class_counts={"pneumonia": 3, "normal": 1000})
    expect(v_size_fail.status == "fail", "rarest class n=3 < 5 → fail")
    v_size_warn = stats.check_sample_size(class_counts={"pneumonia": 7, "normal": 1000})
    expect(v_size_warn.status == "needs_human",
           "rarest class n=7 within (5, 10) → needs_human")
    v_size_ok = stats.check_sample_size(class_counts={"pneumonia": 200, "normal": 1000})
    expect(v_size_ok.status == "pass", "rarest class n=200 → pass")
    v_size_epv = stats.check_sample_size(
        class_counts={"pneumonia": 50, "normal": 1000}, n_predictors=10,
    )
    expect(v_size_epv.status == "fail",
           "EPV rule: rarest=50 < 10*10 -> fail")

    # ------------------------------------------------------------------
    # 5) Pitfalls — direct handlers on synthetic state.
    # ------------------------------------------------------------------
    section("Pitfalls: per-id verdicts on synthetic states")
    state_clean_pitfall = {
        "split_strategy": "patient",
        "multi_source": True,
        "sites": ["NIH", "PadChest"],
        "site_stratified": True,
        "claims_generalization": True,
        "external_validation_datasets": ["PadChest"],
        "test_time_augmentation": False,
        "hypothesized_effects": {"training_size": "positive"},
        "observed_effects": {"training_size": {"sign": "positive", "value": 0.04}},
        "claims_novelty": False,
        "class_counts_train": {"pneumonia": 5000, "normal": 80000},
        "class_counts_test": {"pneumonia": 200, "normal": 1500},
    }
    pf = pitfalls.run_pitfalls(state_clean_pitfall, domain="medical_imaging")
    by_id = {v.check_id: v for v in pf}
    expect(by_id["patient_level_split"].status == "pass",
           "patient_level_split passes when strategy='patient'")
    expect(by_id["site_stratification"].status == "pass",
           "site_stratification passes when site_stratified=True")
    expect(by_id["external_validation_present"].status == "pass",
           "external_validation_present passes with claim + dataset")
    expect(by_id["no_test_time_augmentation_in_baseline"].status == "pass",
           "TTA off → pitfall passes")
    expect(by_id["counterintuitive_signs_flagged"].status == "pass",
           "matching directions → pass")
    expect(by_id["class_balance_reported"].status == "pass",
           "class_balance_reported passes with both train+test counts")

    # Inverted sign → fail
    state_inverted = {
        **state_clean_pitfall,
        "observed_effects": {"training_size": {"sign": "negative", "value": -0.02}},
    }
    pf_inv = pitfalls.run_pitfalls(state_inverted, domain="medical_imaging")
    inv = next(v for v in pf_inv if v.check_id == "counterintuitive_signs_flagged")
    expect(inv.status == "fail",
           "counterintuitive sign inversion → fail")

    # TTA without parity → needs_human
    state_tta_no_match = {
        **state_clean_pitfall,
        "test_time_augmentation": True,
        "tta_baseline_match": False,
    }
    pf_tta = pitfalls.run_pitfalls(state_tta_no_match, domain="medical_imaging")
    tta_v = next(
        v for v in pf_tta if v.check_id == "no_test_time_augmentation_in_baseline"
    )
    expect(tta_v.status == "needs_human",
           "TTA on method without baseline parity → needs_human")

    # Image-level split → fail
    state_image_split = {**state_clean_pitfall, "split_strategy": "image_level"}
    pf_img = pitfalls.run_pitfalls(state_image_split, domain="medical_imaging")
    pls = next(v for v in pf_img if v.check_id == "patient_level_split")
    expect(pls.status == "fail", "image_level split → fail")

    # Multi-source without stratification → fail
    state_no_strat = {**state_clean_pitfall, "site_stratified": False}
    pf_ns = pitfalls.run_pitfalls(state_no_strat, domain="medical_imaging")
    ss = next(v for v in pf_ns if v.check_id == "site_stratification")
    expect(ss.status == "fail",
           "multi-source w/o stratification or per-site metrics → fail")

    # ------------------------------------------------------------------
    # 6) Orchestrator — outcome aggregation.
    # ------------------------------------------------------------------
    section("Orchestrator: outcome aggregation")
    rep_clean = run_all({
        **state_clean_pitfall,
        "train_ids": ["p1", "p2", "p3"],
        "test_ids": ["p4", "p5"],
        "val_ids": ["p6"],
        "baseline_claims": [
            {
                "name": "CheXNet", "dataset": "NIH", "metric": "AUROC",
                "published_value": 0.768, "observed_value": 0.770,
                "tolerance_abs": 0.01,
            },
        ],
    })
    expect(rep_clean.outcome == "clean",
           f"fully clean state → outcome=clean (got {rep_clean.outcome})")

    # Aggregate dedup: only one verdict with the canonical id should
    # contribute to the final report.
    canonical = [v for v in rep_clean.verdicts
                 if v.check_id == "baseline_reproduced_within_tolerance"]
    expect(len(canonical) == 1,
           f"exactly one baseline_reproduced_within_tolerance verdict "
           f"(got {len(canonical)})")
    expect(canonical[0].category == "pitfall",
           "the kept aggregate is the pitfall-category one (TOML severity)")

    rep_block = run_all({**state_clean_pitfall, "split_strategy": "image_level"})
    expect(rep_block.outcome == "block",
           f"patient-level split fail → outcome=block (got {rep_block.outcome})")

    rep_human = run_all({
        **state_clean_pitfall,
        "test_time_augmentation": True,
        "tta_baseline_match": False,
    })
    expect(rep_human.outcome == "needs_human",
           f"TTA-without-parity → outcome=needs_human "
           f"(got {rep_human.outcome})")

    # ------------------------------------------------------------------
    # 7) Checkpoint integration — non-clean opens a stage-4 checkpoint.
    # ------------------------------------------------------------------
    section("Checkpoint integration: non-clean report → stage-4 checkpoint")
    conn = open_db(_DB)
    try:
        run_id = start_run(
            conn,
            project_id="smoke_phase5",
            note="phase 5 verification harness smoke",
        )
        from autoscientist.verify import open_verify_checkpoint

        # Clean report → no checkpoint.
        clean_cp = open_verify_checkpoint(
            conn,
            run_id=run_id,
            from_agent="results_validator",
            to_agent="paper_writer",
            report=rep_clean,
        )
        expect(clean_cp is None, "clean report opens no checkpoint")

        # Block report → stage 4 checkpoint with parsed payload.
        block_cp = open_verify_checkpoint(
            conn,
            run_id=run_id,
            from_agent="results_validator",
            to_agent="paper_writer",
            report=rep_block,
        )
        expect(block_cp is not None, "block report opens a checkpoint")
        rec = cp_manager.get_checkpoint(conn, block_cp)
        assert rec is not None
        expect(rec.stage == 4,
               f"verify checkpoint defaults to stage 4 (got {rec.stage})")
        expect(rec.status == "pending",
               f"new verify checkpoint is pending (got {rec.status})")
        expect(rec.from_agent == "results_validator",
               f"from_agent recorded (got {rec.from_agent!r})")
        parsed = rec.parsed
        expect(parsed is not None and parsed.get("outcome") == "block",
               "parsed payload carries outcome=block")
        expect(isinstance(parsed.get("verdicts"), list)
               and len(parsed["verdicts"]) > 0,
               "parsed payload includes per-verdict detail")

        # needs_human report opens a checkpoint too.
        human_cp = open_verify_checkpoint(
            conn,
            run_id=run_id,
            from_agent="results_validator",
            to_agent="paper_writer",
            report=rep_human,
        )
        rec2 = cp_manager.get_checkpoint(conn, human_cp)
        assert rec2 is not None
        parsed2 = rec2.parsed
        expect(parsed2 is not None and parsed2.get("outcome") == "needs_human",
               "needs_human report serializes correctly")
        conn.commit()
    finally:
        conn.close()

    # ------------------------------------------------------------------
    # 8) Spend invariant.
    # ------------------------------------------------------------------
    section("Spend invariant: zero spend (verify is offline)")
    conn = open_db(_DB)
    try:
        spend = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) AS s FROM budget_ledger"
        ).fetchone()["s"]
        expect(spend == 0.0, f"zero spend (got ${spend})")
    finally:
        conn.close()

    print("\n*** All Phase 5 smoke checks passed. ***")
    print(f"  DB:    {_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
