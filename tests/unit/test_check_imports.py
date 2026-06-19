"""Tests for the static, read-only intra-project import checker (Fix 2).

code_gen has no `execute` tool, so phantom imports (importing a name no sibling
module defines) used to surface only at review. check_imports gives it a static
AST signal. These tests build a tiny sandbox and assert the resolver finds the
phantom import, names the missing symbol, lists what IS available, and ignores
third-party imports.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscientist.tools.check_imports import check_imports


def _sandbox(tmp_path: Path, files: dict[str, str]) -> Path:
    sb = tmp_path / "proj" / "sandbox"
    for rel, content in files.items():
        p = sb / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp_path


def test_flags_phantom_import_and_lists_available(tmp_path):
    root = _sandbox(tmp_path, {
        "src/__init__.py": "",
        "src/config.py": "FOO = 1\nBAZ = 2\n",
        "src/main.py": "from src.config import FOO, BAR\nimport numpy as np\n",
    })
    r = check_imports(project_id="proj", projects_root=root)
    assert r["ok"] is False
    assert len(r["unresolved"]) == 1
    u = r["unresolved"][0]
    assert u["module"] == "src.config"
    assert u["missing"] == ["BAR"]          # FOO resolves, BAR does not
    assert "FOO" in u["available_in_module"] and "BAZ" in u["available_in_module"]


def test_clean_project_is_ok(tmp_path):
    root = _sandbox(tmp_path, {
        "src/__init__.py": "",
        "src/config.py": "STEP = 0.1\n",
        "src/main.py": "from src.config import STEP\nSTEP2 = STEP * 2\n",
    })
    r = check_imports(project_id="proj", projects_root=root)
    assert r["ok"] is True
    assert r["unresolved"] == []


def test_third_party_imports_ignored(tmp_path):
    root = _sandbox(tmp_path, {
        "src/__init__.py": "",
        "src/main.py": "import numpy\nimport torch\nfrom scipy import optimize\n",
    })
    r = check_imports(project_id="proj", projects_root=root)
    assert r["ok"] is True  # none of these resolve to sandbox modules → not checked


def test_missing_module_reported(tmp_path):
    root = _sandbox(tmp_path, {
        "src/__init__.py": "",
        "src/main.py": "from src.nope import thing\n",
    })
    r = check_imports(project_id="proj", projects_root=root)
    assert r["ok"] is False
    assert r["unresolved"][0]["missing"] == ["<module not found>"]


def test_submodule_import_resolves(tmp_path):
    # `from src import config` is valid when src/config.py exists.
    root = _sandbox(tmp_path, {
        "src/__init__.py": "",
        "src/config.py": "X = 1\n",
        "src/main.py": "from src import config\n",
    })
    r = check_imports(project_id="proj", projects_root=root)
    assert r["ok"] is True


def test_build_artifacts_skipped(tmp_path):
    # A stale copy under build/ must not be parsed (it would otherwise resolve
    # against a different src and mask/confuse real errors).
    root = _sandbox(tmp_path, {
        "src/__init__.py": "",
        "src/config.py": "FOO = 1\n",
        "src/main.py": "from src.config import FOO\n",
        "build/lib/src/main.py": "from src.config import GHOST\n",
    })
    r = check_imports(project_id="proj", projects_root=root)
    assert r["ok"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
