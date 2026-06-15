"""Statistical assumption checkers.

Three checks, all stdlib-only:

* **Multicollinearity** — pairwise Pearson correlation on numeric
  features. Any |corr| above the threshold (default 0.95) is flagged.
  We don't compute VIF; that needs a regression solve and adds a numpy
  dependency for marginal gain over pairwise scanning.

* **Normality** — D'Agostino-Pearson-style sanity check via skewness
  and excess kurtosis. We do *not* implement the full K² statistic
  (it needs an inverse-erf for the p-value); instead we flag anything
  with |skew| > ``skew_threshold`` or |excess kurtosis| > ``kurt_threshold``,
  which is the rule-of-thumb that practitioners use (Kline 2016, Bulmer
  1979). Returns ``needs_human`` rather than ``fail`` because some
  legitimate workflows (logistic regression, tree models) don't need
  normality at all.

* **Sample size adequacy** — minimum samples per class plus the EPV
  ("events per variable") rule of thumb (≥10 events per predictor is
  the conventional threshold for stable logistic-regression estimates;
  Peduzzi et al. 1996).

These checks accept their inputs as plain lists/dicts so any agent or
upstream tool can invoke them without first spinning up numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import structlog

from autoscientist.verify.types import Verdict, make_skipped

log = structlog.get_logger("autoscientist.verify.stats")

_DEFAULT_CORR_FAIL = 0.95
_DEFAULT_CORR_WARN = 0.85
_DEFAULT_SKEW = 2.0
_DEFAULT_KURT = 7.0  # excess kurtosis (3 subtracted)
_DEFAULT_MIN_PER_CLASS_FAIL = 5
_DEFAULT_MIN_PER_CLASS_WARN = 10
_DEFAULT_EPV = 10
_DEFAULT_CHANCE_LEVEL = 0.5
_DEFAULT_NEAR_CHANCE_MARGIN = 0.02  # |AUROC - 0.5| <= this counts as near-chance


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _to_floats(xs: Iterable[Any]) -> list[float] | None:
    out: list[float] = []
    for x in xs:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            return None
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _moments(xs: list[float]) -> tuple[float, float, float, float]:
    """Return (mean, var, skew, excess_kurtosis) using sample variance.

    Skewness uses Fisher's definition; excess kurtosis subtracts 3
    so a normal distribution gives 0.
    """
    n = len(xs)
    if n < 2:
        return (xs[0] if xs else 0.0, 0.0, 0.0, 0.0)
    m = _mean(xs)
    diffs = [x - m for x in xs]
    m2 = sum(d * d for d in diffs) / n  # population second moment
    m3 = sum(d ** 3 for d in diffs) / n
    m4 = sum(d ** 4 for d in diffs) / n
    var = sum(d * d for d in diffs) / (n - 1)  # sample variance
    if m2 <= 0.0:
        return (m, var, 0.0, 0.0)
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 * m2) - 3.0
    return (m, var, skew, kurt)


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


# ---------------------------------------------------------------------------
# Public checks
# ---------------------------------------------------------------------------

def check_multicollinearity(
    features: Mapping[str, Iterable[Any]] | None,
    *,
    fail_threshold: float = _DEFAULT_CORR_FAIL,
    warn_threshold: float = _DEFAULT_CORR_WARN,
    severity: str = "fail",
) -> Verdict:
    title = "Multicollinearity (pairwise feature correlation)"
    if not features:
        return make_skipped(
            "multicollinearity", title,
            severity=severity, reason="no features provided",
            category="stats",
        )
    cols: dict[str, list[float]] = {}
    for fname, vals in features.items():
        as_floats = _to_floats(vals)
        if as_floats is None or len(as_floats) < 3:
            continue  # non-numeric or too few points
        cols[fname] = as_floats
    if len(cols) < 2:
        return make_skipped(
            "multicollinearity", title,
            severity=severity,
            reason=f"need ≥2 numeric columns, got {len(cols)}",
            category="stats",
        )
    names = list(cols.keys())
    fails: list[dict[str, Any]] = []
    warns: list[dict[str, Any]] = []
    for i in range(len(names)):
        a = cols[names[i]]
        for j in range(i + 1, len(names)):
            b = cols[names[j]]
            if len(a) != len(b):
                continue
            r = _pearson(a, b)
            ar = abs(r)
            if ar >= fail_threshold:
                fails.append({"a": names[i], "b": names[j], "pearson": round(r, 4)})
            elif ar >= warn_threshold:
                warns.append({"a": names[i], "b": names[j], "pearson": round(r, 4)})
    if fails:
        return Verdict(
            check_id="multicollinearity",
            title=title,
            status="fail",
            severity=severity,  # type: ignore[arg-type]
            detail=f"{len(fails)} pair(s) at |r| ≥ {fail_threshold}",
            evidence={"fails": fails[:25], "warns": warns[:25],
                      "n_features": len(names)},
            category="stats",
        )
    if warns:
        return Verdict(
            check_id="multicollinearity",
            title=title,
            status="needs_human",
            severity=severity,  # type: ignore[arg-type]
            detail=f"{len(warns)} pair(s) at |r| ≥ {warn_threshold} (advisory)",
            evidence={"warns": warns[:25], "n_features": len(names)},
            category="stats",
        )
    return Verdict(
        check_id="multicollinearity",
        title=title,
        status="pass",
        severity=severity,  # type: ignore[arg-type]
        detail=f"all pairs |r| < {warn_threshold}",
        evidence={"n_features": len(names)},
        category="stats",
    )


def check_normality(
    distributions: Mapping[str, Iterable[Any]] | None,
    *,
    skew_threshold: float = _DEFAULT_SKEW,
    kurt_threshold: float = _DEFAULT_KURT,
    severity: str = "needs_human",
) -> Verdict:
    """Per-column skew/kurtosis sanity check.

    ``distributions`` should map a name (e.g. ``residuals``, ``feature_x``)
    to its observed sample. If you don't actually need normality (most
    ML workflows don't), simply omit the key — the check will skip.
    """
    title = "Normality (skew/kurtosis sanity)"
    if not distributions:
        return make_skipped(
            "normality", title,
            severity=severity, reason="no distributions submitted",
            category="stats",
        )
    bad: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    for name, vals in distributions.items():
        as_floats = _to_floats(vals)
        if as_floats is None or len(as_floats) < 8:
            continue  # too few samples to assess
        _, _, skew, kurt = _moments(as_floats)
        row = {
            "name": name,
            "n": len(as_floats),
            "skew": round(skew, 4),
            "excess_kurt": round(kurt, 4),
        }
        inspected.append(row)
        if abs(skew) > skew_threshold or abs(kurt) > kurt_threshold:
            bad.append(row)
    if not inspected:
        return make_skipped(
            "normality", title,
            severity=severity,
            reason="no distribution had ≥8 numeric samples",
            category="stats",
        )
    if bad:
        return Verdict(
            check_id="normality",
            title=title,
            status="needs_human",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"{len(bad)}/{len(inspected)} distribution(s) outside "
                f"|skew|≤{skew_threshold}, |excess kurt|≤{kurt_threshold}"
            ),
            evidence={"violators": bad, "inspected": inspected[:25]},
            category="stats",
        )
    return Verdict(
        check_id="normality",
        title=title,
        status="pass",
        severity=severity,  # type: ignore[arg-type]
        detail=f"all {len(inspected)} distribution(s) within rules-of-thumb",
        evidence={"inspected": inspected[:25]},
        category="stats",
    )


def check_sample_size(
    *,
    class_counts: Mapping[str, int] | None,
    n_predictors: int | None = None,
    min_per_class_fail: int = _DEFAULT_MIN_PER_CLASS_FAIL,
    min_per_class_warn: int = _DEFAULT_MIN_PER_CLASS_WARN,
    epv_threshold: int = _DEFAULT_EPV,
    severity: str = "fail",
) -> Verdict:
    """Sample-size adequacy.

    * Per-class minimum: < ``min_per_class_fail`` → fail; <
      ``min_per_class_warn`` → needs_human.
    * Events-per-variable: when ``n_predictors`` is supplied and the
      task is binary, the rare-class count must be at least
      ``epv_threshold * n_predictors`` to hit the conventional Peduzzi bar.
    """
    title = "Sample size adequacy"
    if not class_counts:
        return make_skipped(
            "sample_size", title,
            severity=severity, reason="class_counts missing",
            category="stats",
        )
    counts = {str(k): int(v) for k, v in class_counts.items()}
    if not counts:
        return make_skipped(
            "sample_size", title,
            severity=severity, reason="class_counts empty",
            category="stats",
        )
    min_n = min(counts.values())
    rarest_class = min(counts, key=counts.__getitem__)
    evidence: dict[str, Any] = {
        "class_counts": counts,
        "min_class": rarest_class,
        "min_n": min_n,
        "min_per_class_fail": min_per_class_fail,
        "min_per_class_warn": min_per_class_warn,
    }
    if n_predictors is not None:
        evidence["n_predictors"] = int(n_predictors)
        evidence["epv_threshold"] = epv_threshold
        evidence["epv_required"] = epv_threshold * int(n_predictors)
    if min_n < min_per_class_fail:
        return Verdict(
            check_id="sample_size",
            title=title,
            status="fail",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"rarest class '{rarest_class}' has n={min_n} "
                f"< {min_per_class_fail}"
            ),
            evidence=evidence,
            category="stats",
        )
    if (
        n_predictors is not None
        and len(counts) == 2
        and min_n < epv_threshold * int(n_predictors)
    ):
        return Verdict(
            check_id="sample_size",
            title=title,
            status="fail",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"events-per-variable below threshold: rarest class n={min_n} "
                f"< {epv_threshold}*{n_predictors}"
            ),
            evidence=evidence,
            category="stats",
        )
    if min_n < min_per_class_warn:
        return Verdict(
            check_id="sample_size",
            title=title,
            status="needs_human",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"rarest class '{rarest_class}' has n={min_n} "
                f"< {min_per_class_warn} (advisory)"
            ),
            evidence=evidence,
            category="stats",
        )
    return Verdict(
        check_id="sample_size",
        title=title,
        status="pass",
        severity=severity,  # type: ignore[arg-type]
        detail=f"all classes ≥ {min_per_class_warn}; rarest n={min_n}",
        evidence=evidence,
        category="stats",
    )


def _coerce_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def check_discrimination_floor(
    aurocs: Iterable[Mapping[str, Any]] | None,
    *,
    chance_level: float = _DEFAULT_CHANCE_LEVEL,
    near_margin: float = _DEFAULT_NEAR_CHANCE_MARGIN,
    severity: str = "needs_human",
) -> Verdict:
    """Near-chance / discrimination-floor guard for AUROC-style metrics.

    This is the guard the 2026-05-31 audit found missing. The existing harness
    only compared observed-vs-published (``baseline_repro``); nothing inspected
    the *absolute* discrimination floor, so the near-chance NIH->PadChest
    transfer (AUROC 0.36-0.55, CIs straddling 0.5) was caught by a human, not
    the system. An AUROC indistinguishable from a coin flip cannot support a
    "the model learned something" / "beats baseline" claim, regardless of how
    cleanly it reproduced an *in-domain* baseline.

    Each entry is a mapping with a point estimate (``point_estimate`` /
    ``auroc`` / ``value``) and optional ``ci_lower`` / ``ci_upper``. Set
    ``primary: false`` on a metric that should be inspected but not escalate
    (e.g. a secondary endpoint). Verdicts:

      * **fail**        — a primary AUROC's whole CI is below ``chance_level``
                          (worse than random → likely label/orientation bug).
      * **needs_human** — a primary AUROC's CI includes ``chance_level``, or its
                          point estimate is within ``near_margin`` of it
                          (not distinguishable from chance → not interpretable).
      * **pass**        — all primary AUROCs are clearly above chance.

    Default ``severity="needs_human"`` so even a ``fail`` status surfaces a
    checkpoint rather than hard-blocking — a near-chance/negative result is
    legitimate to *report*, but must be human-interpreted before any positive
    claim. A domain can pass ``severity="fail"`` to make worse-than-chance a
    hard block.
    """
    title = "Discrimination floor (AUROC vs chance)"
    if not aurocs:
        return make_skipped(
            "discrimination_floor", title,
            severity=severity, reason="no auroc_results provided",
            category="stats",
        )
    near: list[dict[str, Any]] = []
    worse: list[dict[str, Any]] = []
    inspected: list[dict[str, Any]] = []
    for entry in aurocs:
        if not isinstance(entry, Mapping):
            continue
        label = str(
            entry.get("label") or entry.get("name") or entry.get("metric") or "auroc"
        )
        point = _coerce_float(
            entry.get("point_estimate", entry.get("auroc", entry.get("value")))
        )
        lo = _coerce_float(entry.get("ci_lower"))
        hi = _coerce_float(entry.get("ci_upper"))
        if point is None and lo is None and hi is None:
            continue
        primary = bool(entry.get("primary", True))
        row = {
            "label": label, "point_estimate": point,
            "ci_lower": lo, "ci_upper": hi, "primary": primary,
        }
        inspected.append(row)
        if not primary:
            continue
        if hi is not None and hi < chance_level:
            worse.append(row)
        elif lo is not None and hi is not None and lo <= chance_level <= hi:
            near.append(row)
        elif point is not None and abs(point - chance_level) <= near_margin:
            near.append(row)
    if not inspected:
        return make_skipped(
            "discrimination_floor", title,
            severity=severity,
            reason="no usable AUROC values (need a point estimate or CI)",
            category="stats",
        )
    n_primary = sum(1 for r in inspected if r["primary"])
    if worse:
        return Verdict(
            check_id="discrimination_floor", title=title, status="fail",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"{len(worse)} primary AUROC(s) lie entirely below chance "
                f"({chance_level}) — worse than random; likely a label/"
                f"orientation bug, not a usable result"
            ),
            evidence={
                "worse_than_chance": worse[:25],
                "near_chance": near[:25],
                "chance_level": chance_level,
                "n_primary": n_primary,
            },
            category="stats",
        )
    if near:
        return Verdict(
            check_id="discrimination_floor", title=title, status="needs_human",
            severity=severity,  # type: ignore[arg-type]
            detail=(
                f"{len(near)} primary AUROC(s) not distinguishable from chance "
                f"(CI includes {chance_level}, or |point-{chance_level}| <= "
                f"{near_margin}) — results not interpretable without explanation"
            ),
            evidence={
                "near_chance": near[:25],
                "chance_level": chance_level,
                "near_margin": near_margin,
                "n_primary": n_primary,
            },
            category="stats",
        )
    return Verdict(
        check_id="discrimination_floor", title=title, status="pass",
        severity=severity,  # type: ignore[arg-type]
        detail=f"all {n_primary} primary AUROC(s) clearly above chance ({chance_level})",
        evidence={"inspected": inspected[:25], "chance_level": chance_level},
        category="stats",
    )


# ---------------------------------------------------------------------------
# State-driven runner
# ---------------------------------------------------------------------------

def run_stats(state: Mapping[str, Any]) -> list[Verdict]:
    """Run all statistical assumption checks against the pipeline state.

    Expected keys (all optional — missing data → skipped):
      * ``features`` (multicollinearity)
      * ``distributions`` (normality)
      * ``class_counts_train`` and optional ``n_predictors`` (sample size)
      * ``auroc_results`` (discrimination floor / near-chance guard)
    """
    verdicts = [
        check_multicollinearity(state.get("features")),
        check_normality(state.get("distributions")),
        check_sample_size(
            class_counts=state.get("class_counts_train"),
            n_predictors=state.get("n_predictors"),
        ),
        check_discrimination_floor(state.get("auroc_results")),
    ]
    log.info(
        "verify.stats.completed",
        n_checks=len(verdicts),
        n_pass=sum(1 for v in verdicts if v.status == "pass"),
        n_fail=sum(1 for v in verdicts if v.status == "fail"),
    )
    return verdicts
