"""Tests for the peer_reviewer empty-handoff rebuild + latex_compile fallback.

paper_writer can derail (loop on a failed ``latex_compile`` and emit empty
content — run_fe002213…, 2026-06-19), handing peer_reviewer ``(no payload)``
and degenerating CP5. These cover the three fixes: the peer_reviewer payload
rebuild, the runner thin-detector, and the latex_compile sandbox fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoscientist.runtime.payload_files import (
    build_peer_reviewer_payload_from_sandbox,
)
from autoscientist.runtime.runner import _is_thin_peer_reviewer_payload
from autoscientist.tools.registry import ToolContext, _load_tex_fallback


def _project(root: Path, pid: str = "p1") -> Path:
    d = root / pid
    (d / "sandbox").mkdir(parents=True)
    (d / "latex" / "paper").mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# build_peer_reviewer_payload_from_sandbox
# ---------------------------------------------------------------------------

def test_rebuild_unwraps_draft_from_paper_writer_output(tmp_path: Path) -> None:
    _project(tmp_path)
    draft_json = json.dumps({"draft": {"title": "T", "abstract": "A"}, "supplementary": {}})
    out = build_peer_reviewer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, draft_text=draft_json,
        plan_text='{"research_question": "RQ"}', validator_summary={"verdict": "advance"},
    )
    assert out is not None
    payload = json.loads(out)
    assert payload["draft"] == {"title": "T", "abstract": "A"}
    assert payload["context"]["plan"] == {"research_question": "RQ"}
    assert payload["context"]["validator_summary"] == {"verdict": "advance"}
    assert "_reconstructed_by_runner" in payload


def test_rebuild_unwraps_sections_shape(tmp_path: Path) -> None:
    _project(tmp_path)
    draft_json = json.dumps({"sections": {"title": "T", "results": "0.91 AUROC"}})
    out = build_peer_reviewer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, draft_text=draft_json,
    )
    payload = json.loads(out)
    assert payload["draft"]["results"] == "0.91 AUROC"


def test_rebuild_falls_back_to_tex_on_disk(tmp_path: Path) -> None:
    d = _project(tmp_path)
    (d / "latex" / "paper" / "paper.tex").write_text(r"\documentclass{article}\begin{document}Hi\end{document}")
    # no draft_text -> must read the .tex
    out = build_peer_reviewer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, draft_text="   ",
    )
    assert out is not None
    payload = json.loads(out)
    assert payload["draft"]["format"] == "latex"
    assert "documentclass" in payload["draft"]["latex_source"]


def test_rebuild_returns_none_when_nothing(tmp_path: Path) -> None:
    _project(tmp_path)  # empty sandbox/latex, no draft
    assert build_peer_reviewer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, draft_text=None,
    ) is None


def test_rebuild_passes_raw_text_when_first_json_is_stray_citation(tmp_path: Path) -> None:
    """A draft truncated at max_tokens parses to a stray nested citation; the
    rebuild must hand the reviewer the raw prose, NOT the citation
    (regression: run_fe002213, 2026-06-19 — peer_reviewer got a Dijkstra1959
    citation as the 'draft' and rejected with 'no manuscript')."""
    _project(tmp_path)
    draft_text = (
        'I will draft the paper. Looking at the results... '
        '{"key": "Dijkstra1959", "title": "A note on two problems", "year": 1959, '
        '"verified": true} ...abstract and methods discussion continues.'
    )
    out = build_peer_reviewer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, draft_text=draft_text,
    )
    payload = json.loads(out)
    # Not the bare citation — the raw text is forwarded so the reviewer sees prose.
    assert "text" in payload["draft"]
    assert "key" not in payload["draft"]  # not the citation object
    assert "Dijkstra" in payload["draft"]["text"]


def test_rebuild_skips_skeleton_tex(tmp_path: Path) -> None:
    d = _project(tmp_path)
    (d / "latex" / "skeleton").mkdir(parents=True, exist_ok=True)
    (d / "latex" / "skeleton" / "skeleton.tex").write_text("SKELETON")
    # only a skeleton exists -> not a usable draft
    assert build_peer_reviewer_payload_from_sandbox(
        project_id="p1", projects_root=tmp_path, draft_text=None,
    ) is None


# ---------------------------------------------------------------------------
# _is_thin_peer_reviewer_payload
# ---------------------------------------------------------------------------

def test_thin_detector() -> None:
    assert _is_thin_peer_reviewer_payload("") is True
    assert _is_thin_peer_reviewer_payload("   ") is True
    assert _is_thin_peer_reviewer_payload("(no payload)") is True       # short, unparseable
    assert _is_thin_peer_reviewer_payload(json.dumps({"draft": {}})) is True
    assert _is_thin_peer_reviewer_payload(json.dumps({"draft": None})) is True
    assert _is_thin_peer_reviewer_payload(
        json.dumps({"draft": {"title": "T", "abstract": "A"}})
    ) is False
    # a long unparseable prose draft is NOT thin
    assert _is_thin_peer_reviewer_payload("x" * 500) is False
    # a LONG draft truncated mid-JSON (first balanced object is a stray
    # citation) must NOT be thin — it's real content, review it as-is.
    truncated = (
        'Drafting the paper. ```json\n{"sections": {"title": "' + "x" * 500
        + '", "references": [{"key": "Dijkstra1959", "year": 1959}]'  # unterminated
    )
    assert _is_thin_peer_reviewer_payload(truncated) is False


# ---------------------------------------------------------------------------
# latex_compile sandbox fallback
# ---------------------------------------------------------------------------

def test_latex_fallback_autofinds_paper_tex(tmp_path: Path) -> None:
    d = _project(tmp_path)
    (d / "latex" / "paper" / "paper.tex").write_text("PAPER SOURCE")
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    assert _load_tex_fallback(ctx, None) == "PAPER SOURCE"


def test_latex_fallback_explicit_path(tmp_path: Path) -> None:
    d = _project(tmp_path)
    (d / "sandbox" / "draft.tex").write_text("SANDBOX DRAFT")
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    assert _load_tex_fallback(ctx, "sandbox/draft.tex") == "SANDBOX DRAFT"


def test_latex_fallback_rejects_traversal(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "secret.tex").write_text("SECRET")
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    # escaping the project dir must not resolve
    assert _load_tex_fallback(ctx, "../secret.tex") is None


def test_latex_fallback_none_when_no_tex(tmp_path: Path) -> None:
    _project(tmp_path)
    ctx = ToolContext(conn=None, project_id="p1", projects_root=tmp_path)
    assert _load_tex_fallback(ctx, None) is None
