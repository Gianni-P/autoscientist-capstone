"""Domain pitfall library.

KICKOFF.md §9 (Phase 5): "Pitfall checks return Pass / Fail /
Needs-human. Fail blocks the pipeline; Needs-human triggers a checkpoint."

This module:

  1. Loads a domain pitfall TOML (e.g.
     ``config/domains/medical_imaging.toml``) into a list of
     :class:`PitfallCheck` definitions — id, title, declared severity,
     description.
  2. Runs each check id through a registered handler that consults the
     pipeline state. Handlers either delegate to other verify modules
     (e.g. baseline reproduction routes through
     ``verify.baseline_repro``) or read state keys directly.
  3. Returns one :class:`Verdict` per check id.

Adding a new pitfall:

    [[checks]]
    id = "my_new_check"
    title = "..."
    severity = "fail"
    description = "..."

…then register a handler with :func:`register_check`. Calls without a
registered handler return ``skipped`` so the operator sees the gap;
the runner does not silently pass.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from autoscientist.verify import baseline_repro
from autoscientist.verify.types import Severity, Verdict, make_skipped

log = structlog.get_logger("autoscientist.verify.pitfalls")


@dataclass(frozen=True)
class PitfallCheck:
    id: str
    title: str
    severity: Severity
    description: str
    # Optional per-check tuning knobs from the domain TOML's ``[checks.params]``
    # sub-table (e.g. ``min_seeds``). Lets a domain raise a deterministic
    # threshold without forking the handler. Defaults to empty (handler default).
    params: Mapping[str, Any] = field(default_factory=dict)


CheckHandler = Callable[[Mapping[str, Any], PitfallCheck], Verdict]

_REGISTRY: dict[str, CheckHandler] = {}


def register_check(check_id: str, handler: CheckHandler) -> None:
    if check_id in _REGISTRY:
        raise ValueError(f"pitfall handler already registered: {check_id}")
    _REGISTRY[check_id] = handler


def get_handler(check_id: str) -> CheckHandler | None:
    return _REGISTRY.get(check_id)


# ---------------------------------------------------------------------------
# TOML loader
# ---------------------------------------------------------------------------

def load_pitfall_config(path: Path) -> list[PitfallCheck]:
    """Load a domain pitfall TOML file.

    The TOML schema is:

        [[checks]]
        id = "..."
        title = "..."
        severity = "fail" | "needs_human" | "warn"
        description = "..."
    """
    with path.open("rb") as f:
        data = tomllib.load(f)
    raw_checks = data.get("checks") or []
    out: list[PitfallCheck] = []
    seen_ids: set[str] = set()
    for raw in raw_checks:
        cid = raw.get("id")
        if not cid:
            raise ValueError(f"pitfall check missing id in {path}: {raw}")
        if cid in seen_ids:
            raise ValueError(f"duplicate pitfall id '{cid}' in {path}")
        seen_ids.add(cid)
        sev = raw.get("severity", "fail")
        if sev not in ("fail", "needs_human", "warn"):
            raise ValueError(f"invalid severity '{sev}' for pitfall '{cid}'")
        params = raw.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"pitfall '{cid}' params must be a table, got {type(params).__name__}")
        out.append(PitfallCheck(
            id=cid,
            title=raw.get("title", cid),
            severity=sev,
            description=str(raw.get("description", "")).strip(),
            params=params,
        ))
    return out


def domain_config_path(domain: str, *, root: Path | None = None) -> Path:
    """Resolve ``config/domains/<domain>.toml`` against the repo root."""
    if root is None:
        from autoscientist.runtime.config import project_root
        root = project_root()
    return root / "config" / "domains" / f"{domain}.toml"


# ---------------------------------------------------------------------------
# Handlers — concrete check implementations.
# Each pulls only what it needs from state; missing → skipped.
# ---------------------------------------------------------------------------

def _v(check: PitfallCheck, status: str, detail: str, evidence: dict[str, Any]) -> Verdict:
    return Verdict(
        check_id=check.id,
        title=check.title,
        status=status,  # type: ignore[arg-type]
        severity=check.severity,
        detail=detail,
        evidence=evidence,
        category="pitfall",
    )


def _check_patient_level_split(state: Mapping[str, Any], check: PitfallCheck) -> Verdict:
    strategy = state.get("split_strategy")
    if strategy is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="split_strategy not declared",
            category="pitfall",
        )
    s = str(strategy).lower()
    if s in {"patient", "patient_level", "subject", "subject_level", "study"}:
        return _v(check, "pass",
                  f"split strategy is '{strategy}'",
                  {"split_strategy": strategy})
    return _v(check, "fail",
              f"split strategy is '{strategy}', not patient/subject-level",
              {"split_strategy": strategy})


def _check_site_stratification(state: Mapping[str, Any], check: PitfallCheck) -> Verdict:
    multi = bool(state.get("multi_source"))
    sites = state.get("sites") or []
    if not multi and len(sites) <= 1:
        return _v(check, "pass",
                  "single-source dataset; stratification not applicable",
                  {"sites": sites, "multi_source": multi})
    site_strat = state.get("site_stratified")
    per_site = bool(state.get("per_site_metrics"))
    site_strat_declared = "site_stratified" in state
    if not site_strat_declared and "per_site_metrics" not in state:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="multi_source=True but neither site_stratified nor per_site_metrics declared",
            category="pitfall",
        )
    if bool(site_strat) or per_site:
        return _v(check, "pass",
                  "multi-source: stratification or per-site reporting present",
                  {"sites": sites, "site_stratified": bool(site_strat),
                   "per_site_metrics": per_site})
    return _v(check, "fail",
              "multi-source dataset without site stratification or per-site metrics",
              {"sites": sites, "site_stratified": False, "per_site_metrics": False})


def _check_external_validation_present(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    claims_gen = bool(state.get("claims_generalization"))
    if not claims_gen:
        return _v(check, "pass",
                  "no generalization claim; external validation not required",
                  {"claims_generalization": False})
    ext = state.get("external_validation_datasets") or []
    if ext:
        return _v(check, "pass",
                  f"external validation present on {len(ext)} dataset(s)",
                  {"external_validation_datasets": list(ext)})
    return _v(check, "fail",
              "generalization claim present but no external validation dataset",
              {"claims_generalization": True, "external_validation_datasets": []})


def _check_no_test_time_aug_in_baseline(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    tta = state.get("test_time_augmentation")
    if tta is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="test_time_augmentation flag not declared",
            category="pitfall",
        )
    if not bool(tta):
        return _v(check, "pass",
                  "no TTA in evaluation; baseline comparison clean",
                  {"test_time_augmentation": False})
    matched = bool(state.get("tta_baseline_match"))
    if matched:
        return _v(check, "pass",
                  "TTA used uniformly across method and baseline",
                  {"test_time_augmentation": True, "tta_baseline_match": True})
    # TTA on method but not on baseline (or unknown) → human review.
    return _v(check, "needs_human",
              "TTA in evaluation without confirmed parity in baseline",
              {"test_time_augmentation": True, "tta_baseline_match": matched})


def _check_counterintuitive_signs(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """If the methodology declared an expected direction and observed
    results invert it, halt regardless of statistical significance.
    KICKOFF.md §10.
    """
    expected: Mapping[str, str] | None = state.get("hypothesized_effects")
    observed: Mapping[str, Mapping[str, Any]] | None = state.get("observed_effects")
    if not expected:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="no hypothesized_effects declared",
            category="pitfall",
        )
    if not observed:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="no observed_effects to compare",
            category="pitfall",
        )
    inversions: list[dict[str, Any]] = []
    compared: list[str] = []
    for var, expected_dir in expected.items():
        if var not in observed:
            continue
        compared.append(var)
        obs = observed[var]
        obs_sign = obs.get("sign")
        if obs_sign is None and "value" in obs:
            v = obs["value"]
            try:
                obs_sign = "positive" if float(v) > 0 else (
                    "negative" if float(v) < 0 else "zero"
                )
            except (TypeError, ValueError):
                obs_sign = None
        if obs_sign is None:
            continue
        exp_norm = str(expected_dir).lower()
        obs_norm = str(obs_sign).lower()
        if exp_norm in {"positive", "+", "up"}:
            exp_norm = "positive"
        elif exp_norm in {"negative", "-", "down"}:
            exp_norm = "negative"
        if obs_norm in {"+", "up"}:
            obs_norm = "positive"
        elif obs_norm in {"-", "down"}:
            obs_norm = "negative"
        if (
            exp_norm in {"positive", "negative"}
            and obs_norm in {"positive", "negative"}
            and exp_norm != obs_norm
        ):
            inversions.append({
                "variable": var,
                "expected": exp_norm,
                "observed": obs_norm,
                "value": obs.get("value"),
                "p_value": obs.get("p_value"),
            })
    if not compared:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="no overlap between hypothesized and observed effects",
            category="pitfall",
        )
    if inversions:
        return _v(check, "fail",
                  f"{len(inversions)} effect direction(s) contradict prior",
                  {"inversions": inversions, "compared": compared})
    return _v(check, "pass",
              f"all {len(compared)} declared effect directions match",
              {"compared": compared})


def _check_baseline_reproduced(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Delegate to the baseline_repro module's aggregate verdict."""
    verdicts = baseline_repro.run_baseline_repro(state)
    aggregate = next(
        (v for v in verdicts if v.check_id == "baseline_reproduced_within_tolerance"),
        None,
    )
    if aggregate is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="baseline_repro produced no aggregate verdict",
            category="pitfall",
        )
    # Re-key the aggregate as a pitfall-category verdict so it shows up
    # under "pitfall" in the report; keep the underlying detail.
    return Verdict(
        check_id=check.id,
        title=check.title,
        status=aggregate.status,
        severity=check.severity,
        detail=aggregate.detail,
        evidence=aggregate.evidence,
        category="pitfall",
    )


def _check_class_balance_reported(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    train = state.get("class_counts_train")
    test = state.get("class_counts_test")
    have_train = isinstance(train, Mapping) and train
    have_test = isinstance(test, Mapping) and test
    if have_train and have_test:
        return _v(check, "pass",
                  "class counts reported for train and test",
                  {"class_counts_train": dict(train), "class_counts_test": dict(test)})
    missing = []
    if not have_train:
        missing.append("train")
    if not have_test:
        missing.append("test")
    return _v(check, "fail",
              f"class balance not reported for: {', '.join(missing)}",
              {"missing": missing})


def _check_multi_seed_reporting(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """KICKOFF.md §8 requires >=3 seeds per condition with variance reported.

    Accepts either ``n_seeds`` (an int) or ``seeds`` (a list of seed
    values). ``report_seed_variance`` defaults to True only when an
    explicit per-condition variance / SD is included in the results
    payload (``per_condition_variance`` or ``results_with_sd``).
    """
    seeds = state.get("seeds")
    n_declared = state.get("n_seeds")
    if seeds is None and n_declared is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="neither n_seeds nor seeds declared",
            category="pitfall",
        )
    n = len(seeds) if isinstance(seeds, (list, tuple, set)) else None
    # bool is a subtype of int — guard so n_seeds=True/False isn't read as 1/0.
    if n is None and isinstance(n_declared, int) and not isinstance(n_declared, bool):
        n = n_declared
    if n is None:
        return _v(check, "fail",
                  f"could not interpret seed count from state "
                  f"(seeds={seeds!r}, n_seeds={n_declared!r})",
                  {"seeds": seeds, "n_seeds": n_declared})
    variance_flag = state.get("report_seed_variance")
    if variance_flag is None:
        # Infer from results payload when not explicitly declared.
        variance_flag = bool(
            state.get("per_condition_variance")
            or state.get("results_with_sd")
        )
    # Minimum seed count is domain-configurable: clinical_tabular and
    # numerical_optimization mandate >= 5, medical_imaging >= 3 (KICKOFF). The
    # threshold comes from the check's params (default 3) so the deterministic
    # gate matches the domain checklist instead of a single hard-coded literal.
    try:
        min_seeds = int(check.params.get("min_seeds", 3))
    except (TypeError, ValueError):
        min_seeds = 3
    if min_seeds < 1:
        min_seeds = 3
    if n < min_seeds:
        return _v(check, "fail",
                  f"only {n} seed(s) used; this domain requires >= {min_seeds}",
                  {"n_seeds": n, "min_seeds": min_seeds,
                   "report_seed_variance": bool(variance_flag)})
    if not variance_flag:
        return _v(check, "fail",
                  f"{n} seeds run but per-condition variance not reported",
                  {"n_seeds": n, "report_seed_variance": False})
    return _v(check, "pass",
              f"{n} seeds with variance reported",
              {"n_seeds": n, "report_seed_variance": True})


def _check_hyperparameter_tuning_split(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Tuning on the test split contaminates the held-out estimate."""
    split = state.get("hyperparameter_tuning_split")
    if split is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="hyperparameter_tuning_split not declared",
            category="pitfall",
        )
    s = str(split).lower().strip()
    safe = {"validation", "val", "dev", "cv", "cross_validation",
            "cross-validation", "none", "default"}
    leaky = {"test", "holdout", "test_set", "held_out", "train"}
    if s in safe:
        return _v(check, "pass",
                  f"tuning split is '{split}'",
                  {"hyperparameter_tuning_split": split})
    if s in leaky:
        return _v(check, "fail",
                  f"tuning split is '{split}' -- contaminates the held-out estimate",
                  {"hyperparameter_tuning_split": split})
    # Unknown label -- punt to the operator rather than silently passing.
    return _v(check, "needs_human",
              f"tuning split '{split}' is not on the recognised list",
              {"hyperparameter_tuning_split": split,
               "recognised_safe": sorted(safe),
               "recognised_leaky": sorted(leaky)})


def _check_weak_label_provenance(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Surface weak-label datasets unless the limitation is acknowledged.

    ``label_provenance`` maps dataset name to one of:
      * ``expert_radiologist``, ``adjudicated``, ``consensus``  -> strong
      * ``nlp_derived``, ``nlp``, ``automated``, ``weak``,
        ``silver_standard``, ``crowdsourced``                  -> weak
      * anything else                                            -> unknown
    """
    prov = state.get("label_provenance")
    if not isinstance(prov, Mapping) or not prov:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="label_provenance not declared",
            category="pitfall",
        )
    strong = {"expert_radiologist", "expert", "adjudicated",
              "consensus", "radiologist", "clinician", "gold_standard"}
    weak = {"nlp_derived", "nlp", "automated", "weak",
            "silver_standard", "silver", "crowdsourced"}
    weak_datasets: list[str] = []
    unknown_datasets: list[dict[str, str]] = []
    for ds, label in prov.items():
        norm = str(label).lower().strip()
        if norm in strong:
            continue
        if norm in weak:
            weak_datasets.append(ds)
        else:
            unknown_datasets.append({"dataset": ds, "provenance": str(label)})
    disclosed = bool(state.get("weak_label_limitation_disclosed"))
    if not weak_datasets and not unknown_datasets:
        return _v(check, "pass",
                  "all label sources are strong / adjudicated",
                  {"label_provenance": dict(prov)})
    if weak_datasets and disclosed:
        return _v(check, "pass",
                  f"weak labels present on {sorted(weak_datasets)} but "
                  f"limitation disclosed",
                  {"weak_datasets": sorted(weak_datasets),
                   "weak_label_limitation_disclosed": True,
                   "label_provenance": dict(prov)})
    return _v(check, "needs_human",
              "weak / unknown label provenance without disclosure",
              {"weak_datasets": sorted(weak_datasets),
               "unknown_provenance": unknown_datasets,
               "weak_label_limitation_disclosed": disclosed,
               "label_provenance": dict(prov)})


def _check_view_projection_documented(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Chest-X-ray-specific: declare and handle PA/AP/lateral mixing.

    Skipped when the cohort is not chest imaging (no ``view_projections``
    declared). Pass if a single projection or filtered/stratified.
    Fail (warn-severity) if multiple projections without handling.
    """
    views = state.get("view_projections")
    if views is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="view_projections not declared",
            category="pitfall",
        )
    if not isinstance(views, (list, tuple, set)):
        return _v(check, "fail",
                  f"view_projections must be a list (got {type(views).__name__})",
                  {"view_projections": views})
    distinct = sorted({str(v).upper() for v in views})
    if len(distinct) <= 1:
        return _v(check, "pass",
                  f"single projection cohort: {distinct}",
                  {"view_projections": distinct})
    handled = bool(
        state.get("view_projection_filtered")
        or state.get("view_projection_stratified")
        or state.get("per_view_metrics")
    )
    if handled:
        return _v(check, "pass",
                  f"multiple projections {distinct} handled "
                  f"(filtered/stratified/per-view)",
                  {"view_projections": distinct,
                   "view_projection_filtered":
                       bool(state.get("view_projection_filtered")),
                   "view_projection_stratified":
                       bool(state.get("view_projection_stratified")),
                   "per_view_metrics":
                       bool(state.get("per_view_metrics"))})
    return _v(check, "fail",
              f"multiple projections {distinct} without filtering or stratification",
              {"view_projections": distinct})


def _check_confidence_intervals_reported(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """When the run claims one method beats another, demand CIs.

    ``comparison_claims`` is a list of comparison descriptors; presence
    means a comparison was claimed. ``confidence_intervals_present``
    should be True when bootstrap / DeLong / SD-over-seeds CIs are part
    of the results payload.
    """
    claims = state.get("comparison_claims")
    if claims is None:
        return make_skipped(
            check.id, check.title,
            severity=check.severity,
            reason="comparison_claims not declared",
            category="pitfall",
        )
    has_claims = bool(claims) if not isinstance(claims, bool) else claims
    if not has_claims:
        return _v(check, "pass",
                  "no comparison claims; CIs not required",
                  {"comparison_claims": list(claims)
                       if isinstance(claims, (list, tuple, set)) else claims})
    cis_present = bool(state.get("confidence_intervals_present"))
    method = state.get("confidence_interval_method")
    if cis_present:
        return _v(check, "pass",
                  f"comparison claims present and CIs reported "
                  f"({method or 'method unspecified'})",
                  {"comparison_claims": list(claims)
                       if isinstance(claims, (list, tuple, set)) else claims,
                   "confidence_interval_method": method})
    return _v(check, "needs_human",
              "comparison claims without confidence intervals or "
              "equivalent statistical test",
              {"comparison_claims": list(claims)
                   if isinstance(claims, (list, tuple, set)) else claims,
               "confidence_intervals_present": False})


# ---------------------------------------------------------------------------
# Clinical-tabular handlers (config/domains/clinical_tabular.toml).
# Same state-key-reading convention as the imaging handlers above:
# a key that isn't declared yields `skipped` so the gap is visible.
# ---------------------------------------------------------------------------

def _check_preprocessing_fit_on_train_only(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """The canonical tabular leak: fitting imputers/scalers/encoders on the
    full data (or test rows) before the split. An sklearn Pipeline makes the
    fit train-fold-scoped by construction.
    """
    scope = state.get("preprocessing_fit_scope")
    in_pipeline = state.get("preprocessing_in_pipeline")
    if scope is None and in_pipeline is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="neither preprocessing_fit_scope nor preprocessing_in_pipeline declared",
            category="pitfall",
        )
    if bool(in_pipeline):
        return _v(check, "pass",
                  "preprocessing fit inside an sklearn Pipeline (train-fold scoped)",
                  {"preprocessing_in_pipeline": True})
    s = str(scope).lower().strip()
    train_scoped = {"train", "train_only", "train_fold", "training", "pipeline", "fold"}
    leaky = {"all", "full", "dataset", "global", "pre_split", "presplit", "whole", "test"}
    if s in train_scoped:
        return _v(check, "pass",
                  f"preprocessing fit scope is '{scope}'",
                  {"preprocessing_fit_scope": scope})
    if s in leaky:
        return _v(check, "fail",
                  f"preprocessing fit on '{scope}' leaks across the split",
                  {"preprocessing_fit_scope": scope})
    return _v(check, "needs_human",
              f"preprocessing fit scope '{scope}' not recognised",
              {"preprocessing_fit_scope": scope})


def _check_target_leakage_features_excluded(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Features recorded at/after the outcome (or proxies of the label) must
    be reviewed and dropped before modelling.
    """
    reviewed = state.get("target_leakage_reviewed")
    leaking = state.get("leaking_features")
    removed = state.get("leaking_features_removed")
    if reviewed is None and leaking is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="target_leakage_reviewed / leaking_features not declared",
            category="pitfall",
        )
    if isinstance(leaking, (list, tuple, set)):
        leaking_list = list(leaking)
    elif leaking:
        leaking_list = [leaking]
    else:
        leaking_list = []
    if leaking_list and not bool(removed):
        return _v(check, "fail",
                  f"{len(leaking_list)} potential target-leaking feature(s) not removed",
                  {"leaking_features": leaking_list, "leaking_features_removed": bool(removed)})
    if bool(reviewed) or bool(removed):
        return _v(check, "pass",
                  "feature provenance reviewed; no unremoved leaking features",
                  {"target_leakage_reviewed": bool(reviewed),
                   "leaking_features": leaking_list,
                   "leaking_features_removed": bool(removed)})
    # No leaking features AND review not addressed either way → nothing to flag.
    if not leaking_list and reviewed is None:
        return _v(check, "pass",
                  "no leaking features declared",
                  {"target_leakage_reviewed": None, "leaking_features": []})
    # Reviewed was explicitly declared falsy (provenance NOT reviewed) with no
    # removals — an empty leaking_features list is not evidence of absence.
    return _v(check, "needs_human",
              "feature provenance not reviewed; cannot confirm absence of target leakage",
              {"target_leakage_reviewed": bool(reviewed), "leaking_features": leaking_list})


def _check_calibration_reported(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """This project's central question is the calibration-vs-discrimination
    trade-off; reporting AUROC without ECE/Brier is the failure mode itself.
    """
    reported = state.get("metrics_reported") or state.get("calibration_metrics_reported")
    ece = state.get("ece_reported")
    brier = state.get("brier_reported")
    if reported is None and ece is None and brier is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="no metrics_reported / ece_reported / brier_reported declared",
            category="pitfall",
        )
    names = (
        {str(m).lower() for m in reported}
        if isinstance(reported, (list, tuple, set)) else set()
    )
    calib_terms = {
        "ece", "expected_calibration_error", "brier", "brier_score",
        "calibration_slope", "calibration_intercept", "calibration",
    }
    found = sorted(names & calib_terms)
    # `is not None and is not False` so a numeric ECE/Brier of 0.0 (a perfectly
    # calibrated model) still counts as "reported" rather than failing on bool(0.0).
    ece_reported = ece is not None and ece is not False
    brier_reported = brier is not None and brier is not False
    if found or ece_reported or brier_reported:
        return _v(check, "pass",
                  "calibration metric(s) reported alongside discrimination",
                  {"calibration_terms_found": found,
                   "ece_reported": bool(ece), "brier_reported": bool(brier)})
    return _v(check, "fail",
              "discrimination reported without any calibration metric (ECE/Brier/slope)",
              {"metrics_reported": sorted(names)})


def _check_missing_data_disclosed(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Clinical tabular data is heavily (often not-at-random) missing; the
    rate and imputation strategy must be disclosed.
    """
    disclosed = state.get("missing_data_disclosed")
    strategy = state.get("imputation_strategy")
    if disclosed is None and strategy is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="missing_data_disclosed / imputation_strategy not declared",
            category="pitfall",
        )
    if bool(disclosed) or strategy:
        return _v(check, "pass",
                  f"missing-data handling disclosed (strategy={strategy or 'declared'})",
                  {"missing_data_disclosed": bool(disclosed), "imputation_strategy": strategy})
    return _v(check, "needs_human",
              "missing-data mechanism / imputation not disclosed",
              {"missing_data_disclosed": False})


def _check_clinical_utility(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Discrimination + calibration still miss decision usefulness; a
    decision-curve / net-benefit (or threshold-specific) analysis strengthens
    the clinical relevance. Warn-severity: annotates, does not block.
    """
    util = state.get("clinical_utility_analysis")
    if util is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="clinical_utility_analysis not declared",
            category="pitfall",
        )
    if util:
        return _v(check, "pass",
                  f"clinical-utility analysis present: {util}",
                  {"clinical_utility_analysis": util})
    return _v(check, "fail",
              "no decision-curve / net-benefit / threshold analysis",
              {"clinical_utility_analysis": util})


# ---------------------------------------------------------------------------
# Numerical-optimization handlers (config/domains/numerical_optimization.toml).
# Same convention as above: a state key that isn't declared yields `skipped`
# so the gap is visible. These target the failure modes of a constrained /
# path-finding descent scheme (the "limited descent" mountain-rescue project):
# a path that silently violates its own slope constraint, a rotation loop that
# never terminates, comparing path lengths across runs that converge to
# *different* endpoints, and calling a heuristic "optimal" with no optimum to
# compare against.
# ---------------------------------------------------------------------------

def _check_constraint_feasibility(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """The central guarantee of a constrained-descent scheme: the produced
    path must satisfy the declared constraint (descent grade <= theta) at
    *every* step. A path that violates its own constraint anywhere is invalid
    regardless of how short it is.
    """
    satisfied = state.get("constraint_satisfied")
    violations = state.get("constraint_violations")
    if satisfied is None and violations is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="neither constraint_satisfied nor constraint_violations declared",
            category="pitfall",
        )
    n_viol: int | None = None
    if isinstance(violations, bool):
        n_viol = 1 if violations else 0
    elif isinstance(violations, (int, float)):
        n_viol = int(violations)
    elif isinstance(violations, (list, tuple, set)):
        n_viol = len(violations)
    if n_viol is not None and n_viol > 0:
        return _v(check, "fail",
                  f"path violates the descent-grade constraint at {n_viol} step(s)",
                  {"constraint_violations": n_viol})
    # An explicit constraint_satisfied=False is authoritative: the producer is
    # asserting the path is infeasible. Without this, a contradictory
    # constraint_violations=0 alongside satisfied=False would reach the pass
    # branch via `n_viol == 0` and green-light an infeasible path.
    if satisfied is not None and not bool(satisfied):
        return _v(check, "fail",
                  "run explicitly reports the descent-grade constraint is not satisfied",
                  {"constraint_satisfied": False,
                   "constraint_violations": n_viol if n_viol is not None else violations})
    if bool(satisfied) or n_viol == 0:
        return _v(check, "pass",
                  "constraint satisfied at every step of the produced path",
                  {"constraint_satisfied": bool(satisfied) if satisfied is not None else None,
                   "constraint_violations": n_viol if n_viol is not None else 0})
    return _v(check, "fail",
              "constraint feasibility not confirmed",
              {"constraint_satisfied": satisfied, "constraint_violations": violations})


def _check_descent_terminates(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Two non-termination risks for a direction-rotating descent: the inner
    'rotate until walkable' loop can spin forever on a steep face without a
    turn cap, and the outer scheme can iterate without reaching the target.
    The original appendix code has an unbounded inner while-loop.
    """
    turn_cap = state.get("rotation_turn_cap")
    converged = state.get("converged")
    hit_cap = state.get("hit_iteration_cap")
    if turn_cap is None and converged is None and hit_cap is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="termination signals (rotation_turn_cap/converged/hit_iteration_cap) not declared",
            category="pitfall",
        )
    # An unbounded rotation loop is a hard fail — it can hang the run.
    if turn_cap is not None and not turn_cap:
        return _v(check, "fail",
                  "rotation loop has no turn cap — can spin indefinitely on steep faces",
                  {"rotation_turn_cap": turn_cap})
    if bool(hit_cap) and not bool(converged):
        return _v(check, "fail",
                  "scheme hit the iteration cap without converging to the target",
                  {"hit_iteration_cap": True, "converged": bool(converged)})
    if converged is None:
        # Loop is bounded, but convergence was never reported — we must not
        # claim "converged to the target". Escalate rather than pass.
        return _v(check, "needs_human",
                  "rotation loop is bounded but convergence was not reported; "
                  "confirm the target was reached",
                  {"rotation_turn_cap": turn_cap, "converged": None})
    if not bool(converged):
        return _v(check, "needs_human",
                  "run did not report convergence; confirm the target was reached",
                  {"converged": False, "rotation_turn_cap": turn_cap})
    return _v(check, "pass",
              "rotation loop bounded and scheme converged to the target",
              {"rotation_turn_cap": turn_cap, "converged": True})


def _check_well_defined_target(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Comparing total path length across strategies (clockwise vs counter-
    clockwise vs either) is only meaningful when every strategy descends to the
    *same* defined target. Without a defined sink, strategies converge to
    different endpoints and the comparison is ill-posed — the flaw the original
    study identified in its own Discussion.
    """
    has_target = state.get("objective_has_defined_target")
    target = state.get("convergence_target")
    comparable = state.get("endpoints_comparable")
    compares_paths = state.get("compares_path_length")
    if has_target is None and target is None and comparable is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="objective_has_defined_target / convergence_target / endpoints_comparable not declared",
            category="pitfall",
        )
    # A declared target is "present" when it is not None — a falsy-but-valid
    # target (elevation 0 / sea level, index 0) must not read as "no target".
    well_defined = bool(has_target) or target is not None
    # Only an issue when path lengths are actually compared across strategies.
    if compares_paths is not None and not bool(compares_paths):
        return _v(check, "pass",
                  "no cross-strategy path-length comparison; shared target not required",
                  {"compares_path_length": False})
    if well_defined and (comparable is None or bool(comparable)):
        return _v(check, "pass",
                  "objective has a defined target; strategy endpoints are comparable",
                  {"objective_has_defined_target": well_defined,
                   "convergence_target": target,
                   "endpoints_comparable": comparable})
    return _v(check, "fail",
              "path lengths compared without a shared defined target — endpoints differ, comparison ill-posed",
              {"objective_has_defined_target": well_defined,
               "endpoints_comparable": comparable})


def _check_gradient_validated(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Hand-coded analytic gradients are a classic silent-error source; they
    must be validated against finite differences (or supplied by autodiff)
    before any descent result is trusted.
    """
    val = state.get("gradient_validation")
    if val is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="gradient_validation not declared",
            category="pitfall",
        )
    s = str(val).lower().strip()
    trusted = {"finite_difference", "finite_differences", "fd", "fd_checked",
               "autodiff", "automatic_differentiation", "analytic_checked",
               "gradcheck"}
    untrusted = {"analytic_unverified", "hand_coded", "none", "unchecked"}
    if s in trusted:
        return _v(check, "pass",
                  f"gradient validated via '{val}'",
                  {"gradient_validation": val})
    if s in untrusted:
        return _v(check, "needs_human",
                  f"gradient '{val}' not validated against finite differences/autodiff",
                  {"gradient_validation": val})
    return _v(check, "needs_human",
              f"gradient_validation '{val}' not on the recognised list",
              {"gradient_validation": val})


def _check_optimality_gap_reported(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """A 'shortest' / 'near-optimal' path claim needs a reference optimum
    (e.g. a grid shortest-path under the same grade constraint) and a reported
    optimality gap; a heuristic path with no optimum to compare against cannot
    be called optimal.
    """
    claims_opt = state.get("claims_optimality")
    ref = state.get("optimum_reference")
    gap = state.get("optimality_gap")
    if claims_opt is None and ref is None and gap is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="claims_optimality / optimum_reference / optimality_gap not declared",
            category="pitfall",
        )
    if claims_opt is not None and not bool(claims_opt):
        return _v(check, "pass",
                  "no optimality claim made; reference optimum not required",
                  {"claims_optimality": False})
    if ref and gap is not None:
        return _v(check, "pass",
                  f"optimality claim backed by reference '{ref}' (gap={gap})",
                  {"optimum_reference": ref, "optimality_gap": gap})
    return _v(check, "needs_human",
              "optimality / shortest-path claim without a reference optimum and gap",
              {"optimum_reference": ref, "optimality_gap": gap})


def _check_step_discretization_sensitivity(
    state: Mapping[str, Any], check: PitfallCheck
) -> Verdict:
    """Fixed step length and rotation increment are discretization choices that
    drive both the feasibility verdict and every path-length number; their
    sensitivity should be reported. Warn-severity: annotates, does not block.
    """
    analyzed = state.get("step_sensitivity_analyzed")
    if analyzed is None:
        return make_skipped(
            check.id, check.title, severity=check.severity,
            reason="step_sensitivity_analyzed not declared",
            category="pitfall",
        )
    if bool(analyzed):
        return _v(check, "pass",
                  "sensitivity to step length / rotation increment analyzed",
                  {"step_sensitivity_analyzed": True})
    return _v(check, "fail",
              "no sensitivity analysis for step length / rotation increment",
              {"step_sensitivity_analyzed": False})


# Register the medical-imaging checks at import time. Calling
# ``register_check`` is idempotent only via the duplicate guard, so we
# wrap it for safe re-import (e.g. under pytest where modules can be
# loaded multiple times in the same process).
def _register_default_handlers() -> None:
    pairs: list[tuple[str, CheckHandler]] = [
        ("patient_level_split", _check_patient_level_split),
        ("site_stratification", _check_site_stratification),
        ("external_validation_present", _check_external_validation_present),
        ("no_test_time_augmentation_in_baseline", _check_no_test_time_aug_in_baseline),
        ("counterintuitive_signs_flagged", _check_counterintuitive_signs),
        ("baseline_reproduced_within_tolerance", _check_baseline_reproduced),
        ("class_balance_reported", _check_class_balance_reported),
        ("multi_seed_reporting", _check_multi_seed_reporting),
        ("hyperparameter_tuning_split", _check_hyperparameter_tuning_split),
        ("weak_label_provenance_disclosed", _check_weak_label_provenance),
        ("view_projection_documented", _check_view_projection_documented),
        ("confidence_intervals_reported", _check_confidence_intervals_reported),
        # clinical_tabular domain
        ("preprocessing_fit_on_train_only", _check_preprocessing_fit_on_train_only),
        ("target_leakage_features_excluded", _check_target_leakage_features_excluded),
        ("calibration_reported", _check_calibration_reported),
        ("missing_data_handling_disclosed", _check_missing_data_disclosed),
        ("clinical_utility_beyond_auroc", _check_clinical_utility),
        # numerical_optimization domain
        ("constraint_feasibility_verified", _check_constraint_feasibility),
        ("descent_terminates", _check_descent_terminates),
        ("well_defined_convergence_target", _check_well_defined_target),
        ("gradient_validated", _check_gradient_validated),
        ("optimality_gap_reported", _check_optimality_gap_reported),
        ("step_discretization_sensitivity", _check_step_discretization_sensitivity),
    ]
    for cid, h in pairs:
        if cid not in _REGISTRY:
            _REGISTRY[cid] = h


_register_default_handlers()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_pitfalls(
    state: Mapping[str, Any],
    *,
    domain: str = "medical_imaging",
    config_path: Path | None = None,
) -> list[Verdict]:
    """Run all pitfalls for ``domain`` against ``state``.

    ``config_path`` overrides the default ``config/domains/<domain>.toml``
    location — useful for tests that inject a fixture TOML.
    """
    path = config_path or domain_config_path(domain)
    if not path.exists():
        log.warning("verify.pitfalls.config_missing", path=str(path), domain=domain)
        return []
    checks = load_pitfall_config(path)
    verdicts: list[Verdict] = []
    for c in checks:
        handler = get_handler(c.id)
        if handler is None:
            verdicts.append(make_skipped(
                c.id, c.title,
                severity=c.severity,
                reason="no handler registered",
                category="pitfall",
            ))
            continue
        try:
            verdicts.append(handler(state, c))
        except Exception as e:  # never let a single check break the harness
            log.warning(
                "verify.pitfalls.handler_error",
                check_id=c.id, error=str(e), error_type=type(e).__name__,
            )
            verdicts.append(Verdict(
                check_id=c.id,
                title=c.title,
                status="error",
                severity=c.severity,
                detail=f"handler crashed: {type(e).__name__}: {e}",
                evidence={},
                category="pitfall",
            ))
    log.info(
        "verify.pitfalls.completed",
        domain=domain, n_checks=len(checks),
        n_pass=sum(1 for v in verdicts if v.status == "pass"),
        n_fail=sum(1 for v in verdicts if v.status == "fail"),
        n_human=sum(1 for v in verdicts if v.status == "needs_human"),
    )

    # Deduplicate baseline_reproduced_within_tolerance — both the
    # pitfall handler and ``baseline_repro.run_baseline_repro`` will
    # produce a verdict under that id when called separately. The
    # orchestrator (``verify.run_all``) is responsible for dropping
    # the duplicate; this function only emits one (the pitfall version)
    # because it never invokes baseline_repro at the top level.
    return verdicts
