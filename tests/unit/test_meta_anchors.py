"""Unit tests for autoscientist.meta.anchors."""

from __future__ import annotations

import json

import pytest

from autoscientist.meta import anchors


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_anchor_set_sorts_and_dedups(tmp_path):
    base = tmp_path / "anchors" / "idea_gen"
    base.mkdir(parents=True)
    _write(base / "b.json", {"anchor_id": "b", "agent": "idea_gen",
                              "input_payload": "i", "expected_summary": "s"})
    _write(base / "a.json", {"anchor_id": "a", "agent": "idea_gen",
                              "input_payload": "i", "expected_summary": "s"})
    aset = anchors.load_anchor_set(tmp_path, "idea_gen")
    assert [a.anchor_id for a in aset] == ["a", "b"]


def test_missing_directory_returns_empty(tmp_path):
    aset = anchors.load_anchor_set(tmp_path, "nonexistent")
    assert len(aset) == 0


def test_duplicate_anchor_id_raises(tmp_path):
    base = tmp_path / "anchors" / "idea_gen"
    base.mkdir(parents=True)
    _write(base / "x.json", {"anchor_id": "dup", "agent": "idea_gen",
                              "input_payload": "i", "expected_summary": "s"})
    _write(base / "y.json", {"anchor_id": "dup", "agent": "idea_gen",
                              "input_payload": "i", "expected_summary": "s"})
    with pytest.raises(ValueError, match="duplicate anchor_id"):
        anchors.load_anchor_set(tmp_path, "idea_gen")


def test_agent_mismatch_raises(tmp_path):
    base = tmp_path / "anchors" / "idea_gen"
    base.mkdir(parents=True)
    _write(base / "x.json", {"anchor_id": "x", "agent": "wrong_agent",
                              "input_payload": "i", "expected_summary": "s"})
    with pytest.raises(ValueError, match="does not match directory"):
        anchors.load_anchor_set(tmp_path, "idea_gen")


def test_required_fields_validated(tmp_path):
    base = tmp_path / "anchors" / "idea_gen"
    base.mkdir(parents=True)
    _write(base / "x.json", {"anchor_id": "x", "agent": "idea_gen"})
    with pytest.raises(ValueError, match="missing required fields"):
        anchors.load_anchor_set(tmp_path, "idea_gen")


def test_strict_false_skips_bad_files(tmp_path):
    base = tmp_path / "anchors" / "idea_gen"
    base.mkdir(parents=True)
    _write(base / "good.json", {"anchor_id": "good", "agent": "idea_gen",
                                 "input_payload": "i", "expected_summary": "s"})
    (base / "bad.json").write_text("{not json", encoding="utf-8")
    aset = anchors.load_anchor_set(tmp_path, "idea_gen", strict=False)
    assert [a.anchor_id for a in aset] == ["good"]


def test_write_anchor_round_trip(tmp_path):
    a = anchors.Anchor(
        anchor_id="rt", agent="idea_gen",
        input_payload="hello", expected_summary="world",
        expected_keys=("a", "b.c"),
        notes="n",
    )
    anchors.write_anchor(tmp_path, a)
    aset = anchors.load_anchor_set(tmp_path, "idea_gen")
    loaded = aset.by_id("rt")
    assert loaded is not None
    assert loaded.input_payload == "hello"
    assert loaded.expected_keys == ("a", "b.c")
    # Re-write fails without overwrite=True.
    with pytest.raises(FileExistsError):
        anchors.write_anchor(tmp_path, a)
    anchors.write_anchor(tmp_path, a, overwrite=True)


# -- expected_keys path walker -----------------------------------------------


def _a(keys):
    return anchors.Anchor(anchor_id="x", agent="x", input_payload="",
                          expected_summary="", expected_keys=tuple(keys))


def test_has_expected_keys_simple_path():
    ok, missing = anchors.has_expected_keys(_a(["a.b"]), {"a": {"b": 1}})
    assert ok and missing == []


def test_has_expected_keys_missing_path():
    ok, missing = anchors.has_expected_keys(_a(["a.b"]), {"a": {}})
    assert not ok and missing == ["a.b"]


def test_has_expected_keys_list_each():
    ok, missing = anchors.has_expected_keys(
        _a(["ideas[].title"]),
        {"ideas": [{"title": "x"}, {"title": "y"}]},
    )
    assert ok and missing == []
    ok, missing = anchors.has_expected_keys(
        _a(["ideas[].title"]),
        {"ideas": [{"title": "x"}, {"summary": "y"}]},
    )
    assert not ok


def test_has_expected_keys_empty_list_fails():
    ok, missing = anchors.has_expected_keys(
        _a(["ideas[].title"]), {"ideas": []},
    )
    assert not ok and missing == ["ideas[].title"]


def test_has_expected_keys_handles_repeated_segment():
    """Regression: parts.index() previously found only the first match
    when the same segment name appeared more than once in a path."""
    ok, _ = anchors.has_expected_keys(
        _a(["a.a"]), {"a": {"a": "v"}},
    )
    assert ok
