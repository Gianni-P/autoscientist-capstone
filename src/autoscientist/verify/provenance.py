"""Provenance & claim-verification gate.

Implements the "no number without a source" discipline (cf. data-to-paper's
backward-traceable results and EviBound's verification gate): every
quantitative claim in the paper should trace to a value that actually exists in
the materialised ``results`` artifacts.

The paper_writer agent emits a ``provenance`` manifest alongside its draft — a
list of ``{claim, value, source_file, source_key}`` entries, one per
quantitative claim. Two deterministic checks consume it:

* :func:`check_provenance_entries` — for each manifest entry, confirm the cited
  ``source_file`` exists in ``results`` and contains a number matching
  ``value`` within tolerance. A claimed number absent from its cited source is
  a hard ``fail`` (a fabricated or mis-attributed figure).
* :func:`check_claim_coverage` — scan the paper body for numeric tokens and
  flag those not covered by any provenance entry (excluding an allowlist of
  structural numbers — years, small indices, common statistical constants, and
  numbers already declared in the plan). Uncovered numbers are ``needs_human``:
  candidate unbacked claims for the operator to inspect.

Both return ``skipped`` when their inputs are absent, so they are safe in the
standard :func:`autoscientist.verify.run_all` sweep at any stage (the manifest
and paper text only exist post-paper_writer).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import structlog

from autoscientist.verify.types import Verdict, make_skipped

log = structlog.get_logger("autoscientist.verify.provenance")

_CATEGORY = "provenance"

# LaTeX/typographic glue that appears inside numbers: \, (thin space), \; \: \!
# and ~ (non-breaking space). Strip before tokenising.
_LATEX_GLUE_RE = re.compile(r"\\[,;:! ]|~")
_NUM_TOKEN_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")

# Numbers that carry structural / statistical meaning, not an empirical claim.
_STRUCTURAL_CONSTANTS = frozenset({
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    0.05, 0.01, 0.1, 0.5, 0.9, 0.95, 0.99,
    90.0, 95.0, 99.0, 100.0, 1000.0,
})


def _strict_float(x: Any) -> float | None:
    """Parse a number with no substring fallback — ``"E1"`` is NOT ``1``."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = _LATEX_GLUE_RE.sub("", x)
        s = s.replace("$", "").replace("%", "").replace(",", "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _to_float(x: Any) -> float | None:
    """Lenient parse for manifest *values* (``"$6.40$"``, ``"45%"``)."""
    f = _strict_float(x)
    if f is not None:
        return f
    if isinstance(x, str):
        m = _NUM_TOKEN_RE.search(_LATEX_GLUE_RE.sub("", x))
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


def _iter_numbers(obj: Any) -> Iterator[float]:
    """Yield every numeric leaf in a JSON-ish object (strict string parsing)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, str):
        f = _strict_float(obj)
        if f is not None:
            yield f
    elif isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_numbers(v)
    elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for v in obj:
            yield from _iter_numbers(v)


def _matches(a: float, b: float, *, rel_tol: float, abs_tol: float) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * abs(b))


def _resolve_source(results: Any, source_file: Any) -> Any | None:
    """Best-effort match of a manifest ``source_file`` to a results record."""
    if results is None:
        return None
    if not isinstance(results, Mapping):
        # list / single record: nothing to key on; use the whole thing
        return results
    if not source_file:
        return None
    sf = str(source_file).strip()
    if sf in results:
        return results[sf]
    base = sf.rsplit("/", 1)[-1].lower()
    for k, v in results.items():
        kl = str(k).lower()
        if kl == sf.lower() or kl.endswith("/" + base) or kl.rsplit("/", 1)[-1] == base:
            return v
    # loose: the basename appears anywhere in the key (or vice-versa)
    for k, v in results.items():
        kl = str(k).lower()
        if base and (base in kl or kl in sf.lower()):
            return v
    return None


def check_provenance_entries(
    provenance: Any, results: Any, *,
    rel_tol: float = 1e-3, abs_tol: float = 1e-6, severity: str = "fail",
) -> Verdict:
    title = "Paper numbers trace to their cited results source"
    entries: list[Any] = []
    if isinstance(provenance, Mapping):
        entries = list(provenance.values())
    elif isinstance(provenance, Iterable) and not isinstance(provenance, (str, bytes)):
        entries = list(provenance)
    if not entries:
        return make_skipped(
            "provenance_entries", title,
            severity=severity, reason="no provenance manifest emitted",
            category=_CATEGORY,
        )
    results = results if results is not None else {}
    n_ok = 0
    mismatches: list[dict[str, Any]] = []
    missing_source: list[dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, Mapping):
            mismatches.append({"entry": repr(e)[:160], "reason": "entry is not an object"})
            continue
        claim = e.get("claim") or e.get("text") or ""
        value = _to_float(e.get("value"))
        source_file = e.get("source_file") or e.get("source") or e.get("file")
        if value is None:
            mismatches.append({
                "claim": str(claim)[:160], "value": e.get("value"),
                "reason": "value is not numeric",
            })
            continue
        src = _resolve_source(results, source_file)
        if src is None:
            missing_source.append({
                "claim": str(claim)[:160], "value": value,
                "source_file": source_file,
                "reason": "cited source not found in results",
            })
            continue
        if any(_matches(n, value, rel_tol=rel_tol, abs_tol=abs_tol)
               for n in _iter_numbers(src)):
            n_ok += 1
        else:
            mismatches.append({
                "claim": str(claim)[:160], "value": value,
                "source_file": source_file,
                "reason": "value not found in cited source",
            })
    evidence = {
        "n_entries": len(entries), "n_ok": n_ok,
        "n_mismatch": len(mismatches), "n_missing_source": len(missing_source),
        "mismatches": mismatches[:20], "missing_source": missing_source[:20],
    }
    if mismatches or missing_source:
        return Verdict(
            check_id="provenance_entries", title=title,
            status="fail", severity=severity,  # type: ignore[arg-type]
            detail=(
                f"{len(mismatches) + len(missing_source)}/{len(entries)} claimed "
                "number(s) do not trace to their cited results source"
            ),
            evidence=evidence, category=_CATEGORY,
        )
    return Verdict(
        check_id="provenance_entries", title=title,
        status="pass", severity=severity,  # type: ignore[arg-type]
        detail=f"all {len(entries)} provenance entr(y/ies) trace to results",
        evidence=evidence, category=_CATEGORY,
    )


def _plan_numbers(plan: Any) -> set[float]:
    nums: set[float] = set()
    if plan is not None:
        for n in _iter_numbers(plan):
            nums.add(round(n, 6))
    return nums


def _extract_paper_numbers(text: str) -> list[float]:
    cleaned = _LATEX_GLUE_RE.sub("", text)
    out: list[float] = []
    for m in _NUM_TOKEN_RE.findall(cleaned):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _is_structural(n: float) -> bool:
    if n in _STRUCTURAL_CONSTANTS:
        return True
    # integer indices / small counts (sections, figures, tables, small n)
    if float(n).is_integer() and 0 <= n <= 20:
        return True
    # plausible publication years
    if float(n).is_integer() and 1900 <= n <= 2099:
        return True
    return False


def check_claim_coverage(
    paper_text: Any, provenance: Any, *,
    plan: Any = None, rel_tol: float = 1e-3, abs_tol: float = 1e-6,
    severity: str = "needs_human", max_report: int = 20,
) -> Verdict:
    title = "Paper quantitative claims are covered by provenance"
    if not isinstance(paper_text, str) or not paper_text.strip():
        return make_skipped(
            "claim_coverage", title,
            severity=severity, reason="no paper text available",
            category=_CATEGORY,
        )
    # Covered set: plan-declared numbers ∪ provenance values.
    covered: set[float] = set(_plan_numbers(plan))
    entries: list[Any] = []
    if isinstance(provenance, Mapping):
        entries = list(provenance.values())
    elif isinstance(provenance, Iterable) and not isinstance(provenance, (str, bytes)):
        entries = list(provenance)
    for e in entries:
        if isinstance(e, Mapping):
            v = _to_float(e.get("value"))
            if v is not None:
                covered.add(round(v, 6))

    nums = _extract_paper_numbers(paper_text)
    uncovered: list[float] = []
    seen: set[float] = set()
    for n in nums:
        if _is_structural(n):
            continue
        key = round(n, 6)
        if key in seen:
            continue
        if any(_matches(n, c, rel_tol=rel_tol, abs_tol=abs_tol) for c in covered):
            continue
        seen.add(key)
        uncovered.append(n)

    evidence = {
        "n_numbers_in_paper": len(nums),
        "n_covered_values": len(covered),
        "n_uncovered": len(uncovered),
        "uncovered_sample": uncovered[:max_report],
    }
    if uncovered:
        return Verdict(
            check_id="claim_coverage", title=title,
            status="needs_human", severity=severity,  # type: ignore[arg-type]
            detail=(
                f"{len(uncovered)} number(s) in the paper are not backed by a "
                "provenance entry or the plan — inspect for unbacked claims"
            ),
            evidence=evidence, category=_CATEGORY,
        )
    return Verdict(
        check_id="claim_coverage", title=title,
        status="pass", severity=severity,  # type: ignore[arg-type]
        detail="all non-structural numbers in the paper are covered by provenance/plan",
        evidence=evidence, category=_CATEGORY,
    )


def run_provenance(state: Mapping[str, Any]) -> list[Verdict]:
    """Run provenance / claim-verification checks against the pipeline state.

    Expected keys:
      * ``provenance`` — the paper_writer manifest (list of
        ``{claim, value, source_file, source_key}`` or a dict of them)
      * ``results`` — materialised results to trace claimed numbers to
      * ``paper_text`` — the draft body (LaTeX or prose); also accepts
        ``paper_tex`` / ``draft_text``
      * ``plan`` — optional, to allowlist plan-declared constants in coverage
    """
    provenance = state.get("provenance")
    results = state.get("results")
    paper_text = (
        state.get("paper_text")
        or state.get("paper_tex")
        or state.get("draft_text")
    )
    plan = state.get("plan")
    verdicts = [
        check_provenance_entries(provenance, results),
        check_claim_coverage(paper_text, provenance, plan=plan),
    ]
    log.info(
        "verify.provenance.completed",
        statuses=[v.status for v in verdicts],
    )
    return verdicts
