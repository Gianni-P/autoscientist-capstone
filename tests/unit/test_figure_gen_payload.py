"""figure_gen payload reconstruction + the figure manifest collector.

Two reconstruction paths gain behaviour with figure_gen:
  * ``build_figure_gen_payload_from_sandbox`` rebuilds figure_gen's INPUT
    ({plan, results, validator_summary}) from the run's plan + the result JSON
    on disk — the same protection paper_writer used to need from
    results_validator, now that results_validator hands forward to figure_gen.
  * ``build_paper_writer_payload_from_sandbox`` is extended to surface the
    rendered figures (from ``figures/figures.json``, or a disk scan) so
    paper_writer can embed them even after a thin handoff from figure_gen.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoscientist.runtime.payload_files import (
    _collect_figures,
    build_figure_gen_payload_from_sandbox,
    build_paper_writer_payload_from_sandbox,
)
from autoscientist.runtime.runner import _is_thin_figure_gen_payload
from autoscientist.tools.latex import _cache_sha


def _make_results_sandbox(root: Path, project_id: str = "p1") -> Path:
    runs = root / project_id / "sandbox" / "runs" / "validator_run"
    runs.mkdir(parents=True)
    (runs / "e1_summary.json").write_text(json.dumps({
        "experiment": "E1",
        "n_trials": 40,
        "terrain_summaries": [
            {"terrain": "elliptic_paraboloid", "mean_qb": 0.0161, "max_qb": 0.082},
            {"terrain": "monkey_saddle", "mean_qb": 8.05e-05, "max_qb": 0.00017},
        ],
        "internal_validity_passed": False,
    }))
    return root / project_id / "sandbox"


def _add_figures(sandbox: Path, *, with_manifest: bool = True) -> None:
    figs = sandbox / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (figs / "fig1_methods.png").write_bytes(b"\x89PNG\r\n")  # a real (tiny) file on disk
    (figs / "fig2_curve.png").write_bytes(b"\x89PNG\r\n")
    if with_manifest:
        (figs / "figures.json").write_text(json.dumps([
            {"path": "figures/fig1_methods.png", "caption": "Methods.", "label": "fig:methods"},
            {"path": "figures/fig2_curve.png", "caption": "Curve.", "label": "fig:curve"},
            # An entry whose image does NOT exist on disk must be dropped.
            {"path": "figures/ghost.png", "caption": "ghost", "label": "fig:ghost"},
        ]))


# ---------------------------------------------------------------------------
# _is_thin_figure_gen_payload (shares the paper_writer results-thinness rule)
# ---------------------------------------------------------------------------

def test_thin_figure_gen_payload() -> None:
    assert _is_thin_figure_gen_payload("") is True
    assert _is_thin_figure_gen_payload(json.dumps({"plan": {"rq": "x"}, "results": {}})) is True
    not_thin = json.dumps({
        "plan": {"rq": "x"},
        "results": {"terrain_summaries": [{"terrain": "t", "mean_qb": 0.01}]},
    })
    assert _is_thin_figure_gen_payload(not_thin) is False


# ---------------------------------------------------------------------------
# build_figure_gen_payload_from_sandbox
# ---------------------------------------------------------------------------

def test_build_figure_gen_returns_none_without_results(tmp_path: Path) -> None:
    (tmp_path / "p1" / "sandbox").mkdir(parents=True)
    assert build_figure_gen_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, plan_text="plan",
    ) is None


def test_build_figure_gen_carries_results_and_plan(tmp_path: Path) -> None:
    _make_results_sandbox(tmp_path)
    raw = build_figure_gen_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path,
        plan_text=json.dumps({"research_question": "RQ?"}),
        validator_summary={"verdict": "advance"},
    )
    assert raw is not None
    payload = json.loads(raw)
    assert payload["plan"]["research_question"] == "RQ?"
    summ = payload["results"]["validator_run/e1_summary.json"]
    assert summ["n_trials"] == 40
    assert payload["validator_summary"]["verdict"] == "advance"
    assert "_reconstructed_by_runner" in payload
    # figure_gen runs BEFORE figures exist — its rebuilt input must not carry a
    # figures key (that is paper_writer's concern).
    assert "figures" not in payload


# ---------------------------------------------------------------------------
# _collect_figures
# ---------------------------------------------------------------------------

def test_collect_figures_none_without_dir(tmp_path: Path) -> None:
    sandbox = tmp_path / "p1" / "sandbox"
    sandbox.mkdir(parents=True)
    assert _collect_figures(sandbox) == []


def test_collect_figures_from_manifest(tmp_path: Path) -> None:
    sandbox = _make_results_sandbox(tmp_path)
    _add_figures(sandbox, with_manifest=True)
    figs = _collect_figures(sandbox)
    paths = [f["path"] for f in figs]
    assert paths == ["figures/fig1_methods.png", "figures/fig2_curve.png"]  # ghost dropped
    assert figs[0]["caption"] == "Methods."
    assert figs[0]["label"] == "fig:methods"


def test_collect_figures_fallback_scan(tmp_path: Path) -> None:
    sandbox = _make_results_sandbox(tmp_path)
    _add_figures(sandbox, with_manifest=False)  # images but no manifest
    figs = _collect_figures(sandbox)
    paths = sorted(f["path"] for f in figs)
    assert paths == ["figures/fig1_methods.png", "figures/fig2_curve.png"]
    assert all(f["caption"] == "" for f in figs)


# ---------------------------------------------------------------------------
# build_paper_writer_payload_from_sandbox now surfaces figures
# ---------------------------------------------------------------------------

def test_paper_writer_payload_includes_figures(tmp_path: Path) -> None:
    sandbox = _make_results_sandbox(tmp_path)
    _add_figures(sandbox, with_manifest=True)
    raw = build_paper_writer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, plan_text="p",
    )
    assert raw is not None
    payload = json.loads(raw)
    fig_paths = [f["path"] for f in payload["figures"]]
    assert fig_paths == ["figures/fig1_methods.png", "figures/fig2_curve.png"]
    # results still present (regression guard on the existing behaviour)
    assert payload["results"]["validator_run/e1_summary.json"]["n_trials"] == 40


def test_paper_writer_payload_figures_empty_when_none(tmp_path: Path) -> None:
    _make_results_sandbox(tmp_path)  # results but no figures dir
    raw = build_paper_writer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, plan_text="p",
    )
    payload = json.loads(raw)
    assert payload["figures"] == []


# ---------------------------------------------------------------------------
# latex compile cache key includes the figure bytes (no stale-PDF on a
# figure-only change with byte-identical .tex)
# ---------------------------------------------------------------------------

def test_latex_cache_sha_includes_figure_digest() -> None:
    tex = (
        "\\documentclass{article}\\usepackage{graphicx}"
        "\\begin{document}\\includegraphics{figures/fig1.png}\\end{document}"
    )
    base = _cache_sha(tex, None)
    figs_a = _cache_sha(tex, "digest-A")
    figs_b = _cache_sha(tex, "digest-B")
    # Same .tex, different figure bytes => different cache key (no stale render).
    assert base != figs_a
    assert figs_a != figs_b
    # Deterministic for identical inputs.
    assert figs_a == _cache_sha(tex, "digest-A")


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
