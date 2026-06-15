"""Leakage detector.

Two distinct families:

1. **Identity overlap.** When the same patient/scan/study appears in
   both train and test splits, the model can memorize identity-correlated
   features. The check is exact: take the symmetric difference of ID
   sets; any non-empty intersection is a hard fail.

2. **Target leakage in tabular features.** A feature that is a near-
   deterministic function of the target (constant offset, perfect
   classification from a single column) usually means the target was
   accidentally encoded into the feature. We flag features whose Pearson
   correlation with the target exceeds ``corr_threshold`` or that achieve
   perfect single-feature classification accuracy on a held-out split.

Both checks are pure-Python — no numpy. Inputs come from upstream agent
output / dataset fetchers as plain lists.

KICKOFF.md §4 #2 — verification > LLM review.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from autoscientist.verify.types import Verdict, make_skipped

log = structlog.get_logger("autoscientist.verify.leakage")

_DEFAULT_CORR_THRESHOLD = 0.99
_DEFAULT_PERFECT_ACC_TOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers — pure-Python stats
# ---------------------------------------------------------------------------

def _to_floats(xs: Iterable[Any]) -> list[float] | None:
    """Best-effort numeric coercion. Returns None if any element fails."""
    out: list[float] = []
    for x in xs:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y, strict=True))
    if sxx <= 0.0 or syy <= 0.0:
        return 0.0
    return sxy / ((sxx ** 0.5) * (syy ** 0.5))


def _single_feature_threshold_acc(feature: list[float], target: list[int]) -> float:
    """Best accuracy from a single threshold split on ``feature``.

    Sweeps every distinct-value boundary as a candidate threshold and
    returns the best accuracy considering both polarities. Returns 0.0
    when the feature is constant (no valid split) or the target has only
    one class. Splits that fall inside a run of identical feature values
    are skipped — they correspond to thresholds that no real classifier
    can place — which prevents the sort tiebreaker (``(value, target)``)
    from spuriously reporting acc=1.0 on constant features.
    """
    n = len(feature)
    if n == 0 or n != len(target):
        return 0.0
    pairs = sorted(zip(feature, target, strict=True))
    n_pos = sum(target)
    if n_pos == 0 or n_pos == n:
        return 0.0
    best = 0.0
    pos_below = 0
    for i, (_, t) in enumerate(pairs):
        if t == 1:
            pos_below += 1
        if i + 1 == n:
            break  # "all below" is the trivial-baseline split, not informative
        if pairs[i][0] == pairs[i + 1][0]:
            continue  # threshold inside a tie run is not realizable
        below = i + 1
        above = n - below
        # polarity A: above is the positive class
        correct_above_pos = n_pos - pos_below
        correct_below_neg = below - pos_below
        acc_a = (correct_above_pos + correct_below_neg) / n
        # polarity B: below is the positive class
        correct_above_neg = above - correct_above_pos
        acc_b = (pos_below + correct_above_neg) / n
        if acc_a > best:
            best = acc_a
        if acc_b > best:
            best = acc_b
    return best


# ---------------------------------------------------------------------------
# Public checks
# ---------------------------------------------------------------------------

def check_id_overlap(
    *,
    train_ids: Iterable[Any] | None,
    test_ids: Iterable[Any] | None,
    val_ids: Iterable[Any] | None = None,
    severity: str = "fail",
) -> Verdict:
    """Hard check: train/val/test ID sets must be pairwise disjoint."""
    title = "Train/val/test identity overlap"
    if train_ids is None or test_ids is None:
        return make_skipped(
            "id_overlap", title,
            severity=severity, reason="train_ids or test_ids missing",
            category="leakage",
        )
    train_set = {str(x) for x in train_ids}
    test_set = {str(x) for x in test_ids}
    val_set = {str(x) for x in (val_ids or [])}
    overlaps: dict[str, list[str]] = {}
    pairs = [("train_test", train_set & test_set)]
    if val_set:
        pairs.append(("train_val", train_set & val_set))
        pairs.append(("val_test", val_set & test_set))
    for name, inter in pairs:
        if inter:
            overlaps[name] = sorted(inter)[:10]  # cap evidence
    if not overlaps:
        return Verdict(
            check_id="id_overlap",
            title=title,
            status="pass",
            severity=severity,  # type: ignore[arg-type]
            detail="splits are pairwise disjoint",
            evidence={
                "n_train": len(train_set),
                "n_test": len(test_set),
                "n_val": len(val_set),
            },
            category="leakage",
        )
    total_overlap = sum(len(v) for v in overlaps.values())
    return Verdict(
        check_id="id_overlap",
        title=title,
        status="fail",
        severity=severity,  # type: ignore[arg-type]
        detail=f"{total_overlap} ID(s) appear in more than one split",
        evidence={
            "overlapping_pairs": overlaps,
            "n_train": len(train_set),
            "n_test": len(test_set),
            "n_val": len(val_set),
        },
        category="leakage",
    )


def check_target_leakage(
    *,
    features: Mapping[str, Iterable[Any]] | None,
    target: Iterable[Any] | None,
    corr_threshold: float = _DEFAULT_CORR_THRESHOLD,
    perfect_acc_tol: float = _DEFAULT_PERFECT_ACC_TOL,
    severity: str = "fail",
) -> Verdict:
    """Flag tabular features that are near-deterministic predictors of the target.

    Two signals:
      * |Pearson(feature, target)| >= ``corr_threshold`` (numeric features)
      * single-feature threshold classifier accuracy >= 1 - ``perfect_acc_tol``
        when the target is binary (encoded as 0/1)
    """
    title = "Target leakage in tabular features"
    if features is None or target is None or not features:
        return make_skipped(
            "target_leakage", title,
            severity=severity, reason="features or target not provided",
            category="leakage",
        )
    target_list = list(target)
    n = len(target_list)
    if n < 4:
        return make_skipped(
            "target_leakage", title,
            severity=severity, reason=f"too few samples to assess (n={n})",
            category="leakage",
        )
    target_floats = _to_floats(target_list)
    target_set = set(target_list)
    is_binary = target_set <= {0, 1}
    target_bin = (
        [int(bool(t)) for t in target_list] if is_binary else None
    )

    suspicious: list[dict[str, Any]] = []
    for fname, fvals in features.items():
        fvals_list = list(fvals)
        if len(fvals_list) != n:
            continue  # silently skip ragged columns; not this check's job to flag
        f_floats = _to_floats(fvals_list)
        flagged: dict[str, Any] = {"feature": fname}
        is_flagged = False
        if f_floats is not None and target_floats is not None:
            corr = _pearson(f_floats, target_floats)
            flagged["pearson"] = round(corr, 6)
            if abs(corr) >= corr_threshold:
                is_flagged = True
                flagged["reason"] = f"|pearson|={abs(corr):.4f}>={corr_threshold}"
        if (
            not is_flagged
            and target_bin is not None
            and f_floats is not None
        ):
            acc = _single_feature_threshold_acc(f_floats, target_bin)
            flagged["single_feature_acc"] = round(acc, 6)
            if acc >= 1.0 - perfect_acc_tol:
                is_flagged = True
                flagged["reason"] = f"single-feature threshold acc={acc:.4f}"
        if is_flagged:
            suspicious.append(flagged)

    if not suspicious:
        return Verdict(
            check_id="target_leakage",
            title=title,
            status="pass",
            severity=severity,  # type: ignore[arg-type]
            detail=f"{len(features)} feature(s) checked; none near-deterministic",
            evidence={"n_features": len(features), "n_samples": n},
            category="leakage",
        )
    return Verdict(
        check_id="target_leakage",
        title=title,
        status="fail",
        severity=severity,  # type: ignore[arg-type]
        detail=f"{len(suspicious)} feature(s) near-deterministically predict the target",
        evidence={
            "suspicious": suspicious[:25],  # cap
            "n_features": len(features),
            "n_samples": n,
            "corr_threshold": corr_threshold,
        },
        category="leakage",
    )


# ---------------------------------------------------------------------------
# State-driven runner
# ---------------------------------------------------------------------------

def run_leakage(state: Mapping[str, Any]) -> list[Verdict]:
    """Run all leakage checks against the pipeline state dict.

    Expected keys (all optional — missing ones cause ``skipped``):
      * ``train_ids``, ``val_ids``, ``test_ids``  — iterables of identity strings
      * ``features``                              — ``dict[str, list]`` tabular features
      * ``target``                                — iterable, target column
    """
    verdicts = [
        check_id_overlap(
            train_ids=state.get("train_ids"),
            test_ids=state.get("test_ids"),
            val_ids=state.get("val_ids"),
        ),
        check_target_leakage(
            features=state.get("features"),
            target=state.get("target"),
        ),
    ]
    log.info(
        "verify.leakage.completed",
        n_checks=len(verdicts),
        n_pass=sum(1 for v in verdicts if v.status == "pass"),
        n_fail=sum(1 for v in verdicts if v.status == "fail"),
    )
    return verdicts
