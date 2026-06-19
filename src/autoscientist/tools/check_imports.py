"""Static, read-only intra-project import consistency checker.

``code_gen`` writes one file at a time and has **no** ``execute`` tool — that
was removed deliberately because giving it ``execute`` made it burn every tool
round in a debug-spin and never hand off (see ``agents/code_gen.py``). The cost
of writing blind is the #1 reason its output is rejected at review: a module
imports a name no sibling module defines, so the code dies with ``ImportError``
before any logic runs (e.g. ``from src.metrics import auroc`` where
``src/metrics.py`` never defines ``auroc``).

This tool closes that gap *without* letting the agent run code. It parses every
``.py`` file in the sandbox with :mod:`ast`, collects the top-level names each
module defines (and re-exports), and resolves every first-party import against
them — reporting the unresolved ones together with the names that *are*
available in the target module, so the agent can fix the import or the source.

Pure AST: it never imports or executes the code, so it cannot spin, hit the
network, or have side effects. Third-party / stdlib imports (numpy, torch, …)
are ignored — only imports that resolve to a file in the sandbox are checked.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger("autoscientist.tools.check_imports")

#: Directories never walked: caches, VCS, and build artifacts. ``build``/``dist``
#: matter here because a stray ``setup.py build`` leaves a stale *copy* of src
#: under ``build/lib/`` that would otherwise be parsed and confuse resolution.
_SKIP_DIRS = frozenset({
    "__pycache__", ".venv", ".git", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "build", "dist", ".eggs", "node_modules",
})

#: Caps so a huge tree can't blow the tool-result payload.
_MAX_AVAILABLE = 40
_MAX_UNRESOLVED = 60


class SandboxEscape(RuntimeError):
    """Raised when ``subdir`` would resolve outside the sandbox."""


def _module_dotted(rel_path: PurePosixPath) -> str:
    """Map a sandbox-relative ``.py`` path to its dotted module name.

    ``src/config.py`` → ``src.config``; ``src/__init__.py`` → ``src``;
    ``main.py`` → ``main``.
    """
    parts = list(rel_path.parts)
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip ".py"
    return ".".join(parts)


def _collect_top_level(tree: ast.Module) -> tuple[set[str], bool]:
    """Return (names defined/re-exported at module top level, has_star_import).

    Imported names count as "available" because ``from m import x`` makes ``x``
    re-importable from ``m``. A ``from x import *`` makes name resolution against
    that module undecidable, signalled by the bool.
    """
    names: set[str] = set()
    has_star = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                _add_target_names(tgt, names)
        elif isinstance(node, ast.AnnAssign):
            _add_target_names(node.target, names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    has_star = True
                else:
                    names.add(alias.asname or alias.name)
    return names, has_star


def _add_target_names(target: ast.expr, names: set[str]) -> None:
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _add_target_names(elt, names)


def check_imports(*, project_id: str, projects_root: Path | str, subdir: str = "") -> dict[str, object]:
    """Resolve every first-party import in the sandbox; report unresolved ones.

    Returns a dict with ``ok`` (no unresolved imports and no syntax errors),
    ``files_checked``, ``unresolved`` (each: file, line, the import statement,
    the target module, the missing names, and what *is* available there),
    ``syntax_errors``, and a human ``summary``.
    """
    sandbox = (Path(projects_root) / project_id / "sandbox")
    rel_sub = PurePosixPath(subdir or "")
    if rel_sub.is_absolute():
        raise SandboxEscape(f"subdir must be relative, got: {subdir}")
    start = (sandbox / rel_sub).resolve()
    sandbox_resolved = sandbox.resolve()
    # Real path-boundary containment (not a string prefix, which would accept a
    # prefix-colliding sibling like ``../sandbox_x``).
    if not start.is_relative_to(sandbox_resolved):
        raise SandboxEscape(f"subdir escapes sandbox: {subdir}")
    if not start.exists():
        return {
            "ok": True, "files_checked": 0, "unresolved": [], "syntax_errors": [],
            "summary": "no sandbox directory to check",
        }

    # --- Pass 1: discover modules, the names each defines, and packages. -----
    defined: dict[str, set[str]] = {}
    has_star: set[str] = set()
    file_for: dict[str, PurePosixPath] = {}  # dotted -> rel path (for reporting)
    all_modules: set[str] = set()
    all_packages: set[str] = set()
    syntax_errors: list[dict[str, object]] = []
    py_files: list[tuple[PurePosixPath, Path]] = []

    for path in sorted(start.rglob("*.py")):
        rel_parts = path.relative_to(sandbox_resolved).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        rel = PurePosixPath(*rel_parts)
        py_files.append((rel, path))
        dotted = _module_dotted(rel)
        all_modules.add(dotted)
        file_for[dotted] = rel
        # Every dotted prefix is an importable package path.
        parts = dotted.split(".")
        for i in range(1, len(parts)):
            all_packages.add(".".join(parts[:i]))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(rel))
        except SyntaxError as e:
            syntax_errors.append({"file": rel.as_posix(), "line": e.lineno or 0, "error": e.msg})
            continue
        names, star = _collect_top_level(tree)
        defined[dotted] = names
        if star:
            has_star.add(dotted)

    # Top-level names that are first-party (a sandbox module or package root).
    first_party_tops = {m.split(".")[0] for m in all_modules}

    def _exists(mod: str) -> bool:
        return mod in all_modules or mod in all_packages

    # --- Pass 2: resolve imports in each file. -------------------------------
    unresolved: list[dict[str, object]] = []

    for rel, path in py_files:
        dotted = _module_dotted(rel)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(rel))
        except SyntaxError:
            continue  # already recorded in pass 1

        # Package containing this module, for resolving relative imports.
        is_pkg_init = rel.name == "__init__.py"
        pkg_parts = dotted.split(".") if is_pkg_init else dotted.split(".")[:-1]

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target_parts = _resolve_from_target(node, pkg_parts)
                if target_parts is None:
                    continue
                top = target_parts[0] if target_parts else ""
                if top not in first_party_tops:
                    continue  # third-party / stdlib — not our problem
                target = ".".join(target_parts)
                stmt = _stmt_text(node)
                if not _exists(target):
                    unresolved.append({
                        "file": rel.as_posix(), "line": node.lineno, "import": stmt,
                        "module": target, "missing": ["<module not found>"],
                        "available_in_module": [],
                    })
                    continue
                if target in has_star:
                    continue  # `from x import *` in target — undecidable, skip
                avail = defined.get(target, set())
                missing = []
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    n = alias.name
                    if n in avail:
                        continue
                    if _exists(f"{target}.{n}"):
                        continue  # importing a submodule of a package
                    missing.append(n)
                if missing:
                    unresolved.append({
                        "file": rel.as_posix(), "line": node.lineno, "import": stmt,
                        "module": target, "missing": missing,
                        "available_in_module": sorted(avail)[:_MAX_AVAILABLE],
                    })
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    full = alias.name
                    top = full.split(".")[0]
                    if top not in first_party_tops:
                        continue
                    if not _exists(full):
                        unresolved.append({
                            "file": rel.as_posix(), "line": node.lineno,
                            "import": f"import {full}", "module": full,
                            "missing": ["<module not found>"], "available_in_module": [],
                        })

    truncated = len(unresolved) > _MAX_UNRESOLVED
    if truncated:
        unresolved = unresolved[:_MAX_UNRESOLVED]
    ok = not unresolved and not syntax_errors

    if ok:
        summary = f"OK — {len(py_files)} file(s) checked, all first-party imports resolve."
    else:
        bits = []
        if unresolved:
            bits.append(f"{len(unresolved)} unresolved import(s)")
        if syntax_errors:
            bits.append(f"{len(syntax_errors)} file(s) with syntax errors")
        extra = " (truncated)" if truncated else ""
        summary = (
            "; ".join(bits) + extra
            + ". Fix the source module to define the missing name, or change the "
            "import to a name that already exists — do NOT import a name nothing defines."
        )

    log.info(
        "check_imports.done", project_id=project_id, files=len(py_files),
        unresolved=len(unresolved), syntax_errors=len(syntax_errors), ok=ok,
    )
    return {
        "ok": ok,
        "files_checked": len(py_files),
        "unresolved": unresolved,
        "syntax_errors": syntax_errors,
        "truncated": truncated,
        "summary": summary,
    }


def _resolve_from_target(node: ast.ImportFrom, pkg_parts: list[str]) -> list[str] | None:
    """Resolve an ``ImportFrom`` to the dotted parts of its target module.

    Handles absolute (``level == 0``) and relative (``level >= 1``) imports.
    Returns ``None`` if a relative import points above the sandbox root.
    """
    if node.level == 0:
        return node.module.split(".") if node.module else None
    # Relative: level 1 == current package, level 2 == parent, …
    up = node.level - 1
    if up > len(pkg_parts):
        return None
    base = pkg_parts[: len(pkg_parts) - up] if up else list(pkg_parts)
    return base + node.module.split(".") if node.module else base


def _stmt_text(node: ast.ImportFrom) -> str:
    """Reconstruct a readable ``from ... import ...`` line for the report."""
    dots = "." * node.level
    mod = node.module or ""
    names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in node.names)
    return f"from {dots}{mod} import {names}"
