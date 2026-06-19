"""citation_check caching: identical re-checks must short-circuit.

paper_writer (stateless between turns) was re-verifying already-verified
references every tool round — 73 calls in one run, Nash2007 x15 — each an
expensive model round (run_fe002213…, 2026-06-19). The handler now caches by
citation content and returns the cached result + an ALREADY CHECKED note on a
repeat, without re-running verification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscientist.state.db import open_db
from autoscientist.tools import citation_check
from autoscientist.tools.registry import ToolContext, _h_citation_check

_CITATION = {
    "key": "Nash2007",
    "title": "Theta*: Any-Angle Path Planning on Grids",
    "authors": ["Nash, A."],
    "year": 2007,
    "doi_or_arxiv": "10.1613/jair.2994",
}


class _FakeChk:
    def __init__(self, verified: bool = True) -> None:
        self.verified = verified

    def to_dict(self) -> dict:
        return {"citation_key": "Nash2007", "verified": self.verified, "confidence": 1.0}


def test_repeat_check_is_cached_and_nudged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = open_db(tmp_path / "c.db")
    ctx = ToolContext(conn=conn, project_id="p1", projects_root=tmp_path)
    calls = {"n": 0}

    def fake_verify(citation, *, conn=None, **kw):
        calls["n"] += 1
        return _FakeChk(verified=True)

    monkeypatch.setattr(citation_check, "verify_citation", fake_verify)

    r1 = _h_citation_check({"citation": _CITATION}, ctx)
    assert r1["verified"] is True
    assert "_note" not in r1
    assert calls["n"] == 1

    # Identical repeat → cached result + ALREADY CHECKED note, NO re-verify.
    r2 = _h_citation_check({"citation": _CITATION}, ctx)
    assert r2["verified"] is True
    assert "ALREADY CHECKED" in r2["_note"]
    assert calls["n"] == 1

    # A different citation is still verified (cache is per-content).
    other = {**_CITATION, "key": "Other2020", "title": "Something Else", "doi_or_arxiv": "10.0/x"}
    _h_citation_check({"citation": other}, ctx)
    assert calls["n"] == 2


def test_unverified_repeat_also_short_circuits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unverifiable citation is also cached — re-checking it is equally
    pointless (it must be dropped, not re-verified)."""
    conn = open_db(tmp_path / "c.db")
    ctx = ToolContext(conn=conn, project_id="p1", projects_root=tmp_path)
    calls = {"n": 0}

    def fake_verify(citation, *, conn=None, **kw):
        calls["n"] += 1
        return _FakeChk(verified=False)

    monkeypatch.setattr(citation_check, "verify_citation", fake_verify)

    bad = {"key": "Fabricated2099", "title": "Nonexistent", "doi_or_arxiv": "10.0/nope"}
    r1 = _h_citation_check({"citation": bad}, ctx)
    assert r1["verified"] is False
    r2 = _h_citation_check({"citation": bad}, ctx)
    assert r2["verified"] is False
    assert "ALREADY CHECKED" in r2["_note"]
    assert calls["n"] == 1  # not re-verified


def test_no_conn_still_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a DB connection (no cache), the handler still verifies."""
    ctx = ToolContext(conn=None)
    monkeypatch.setattr(citation_check, "verify_citation", lambda c, **kw: _FakeChk(True))
    r = _h_citation_check({"citation": _CITATION}, ctx)
    assert r["verified"] is True
