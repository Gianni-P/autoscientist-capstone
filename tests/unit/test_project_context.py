"""Tests for per-project context rendering / injection (Fix 1).

The shared code_gen/test_gen prompts used to hardcode the chest-xray dataset
block, which leaked into every project (including no-dataset numerical ones) and
a weak local model followed it literally. render_project_context now sources the
domain/datasets from the project's own config.toml, and inject_project_context
substitutes the {{PROJECT_CONTEXT}} marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscientist.runtime.project_context import (
    PLACEHOLDER,
    inject_project_context,
    render_project_context,
)


def _project(tmp_path: Path, toml: str) -> Path:
    p = tmp_path / "proj"
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.toml").write_text(toml, encoding="utf-8")
    return tmp_path


def test_no_dataset_project_emits_no_data_directive(tmp_path):
    root = _project(tmp_path, """
[project]
domain = "numerical_optimization"
description = "constrained descent study"
[datasets]
allowed = []
[experiment.defaults]
step_length = 0.1
""")
    block = render_project_context(root, "proj")
    assert "numerical_optimization" in block
    assert "NO external datasets" in block
    assert "ChestX-ray14" not in block          # no contamination
    assert "`step_length` = 0.1" in block        # defaults surfaced


def test_dataset_project_renders_known_layout(tmp_path):
    root = _project(tmp_path, """
[project]
domain = "medical_imaging"
description = "pneumonia detection"
[datasets]
allowed = ["nih_chestxray14", "padchest"]
""")
    block = render_project_context(root, "proj")
    assert "NIH ChestX-ray14" in block and "Data_Entry_2017.csv" in block
    assert "PadChest" in block
    assert "NO external datasets" not in block


def test_unknown_dataset_renders_generic_note(tmp_path):
    root = _project(tmp_path, """
[project]
domain = "clinical_tabular"
description = "tabular risk"
[datasets]
allowed = ["support2"]
""")
    block = render_project_context(root, "proj")
    assert "support2" in block
    assert "fetched in-sandbox" in block


def test_inject_replaces_marker(tmp_path):
    root = _project(tmp_path, """
[project]
domain = "numerical_optimization"
description = "x"
[datasets]
allowed = []
""")
    text = f"Intro.\n\n{PLACEHOLDER}\n\nOutro."
    out = inject_project_context(text, root, "proj")
    assert PLACEHOLDER not in out
    assert "numerical_optimization" in out
    assert out.startswith("Intro.") and out.rstrip().endswith("Outro.")


def test_inject_strips_marker_when_no_config(tmp_path):
    text = f"A\n{PLACEHOLDER}\nB"
    out = inject_project_context(text, tmp_path, "does-not-exist")
    assert PLACEHOLDER not in out


def test_inject_noop_without_marker(tmp_path):
    text = "no marker here"
    assert inject_project_context(text, tmp_path, "proj") == text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
