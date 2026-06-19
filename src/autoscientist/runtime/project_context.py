"""Render a per-project context block injected into agent system prompts.

This replaces the hardcoded, single-domain dataset block that used to live
verbatim in ``prompts/code_gen.md`` (NIH ChestX-ray14 / PadChest paths) and the
medical examples in ``prompts/test_gen.md`` (patient-level split, AUROC). Those
leaked the chest-xray domain into *every* project — including pure-NumPy ones
like ``math693a-limited-descent`` (``[datasets].allowed = []``) — and a weak
local model (qwen3-coder) followed the irrelevant instructions literally,
emitting medical-imaging scaffolding (``MedicalDataset``, ``create_patient_split``,
``auroc``) that could never import against the real optimization code. Sonnet had
the headroom to ignore the contamination; Qwen did not, which is most of why the
code/review loop never converged on the local model (see the run audit, 2026-06-12).

The runner substitutes :data:`PLACEHOLDER` in a prompt's body with the rendered
block at load time (``runtime/runner._drive_loop``), so the domain, objective, and
dataset facts come from the project's own ``config.toml`` instead of being baked
into a shared prompt.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("autoscientist.runtime.project_context")

#: Marker substituted in prompt bodies. Kept ASCII/brace-delimited so it never
#: collides with prose. If a prompt contains it but no project config is found,
#: it is replaced with the empty string (the literal token must never reach the
#: model).
PLACEHOLDER = "{{PROJECT_CONTEXT}}"

#: On-disk layout for the large, operator-pre-staged imaging archives. Rendered
#: ONLY when a project's ``[datasets].allowed`` actually lists the dataset, so a
#: project that doesn't use one never sees its paths. Small/tabular datasets are
#: deliberately absent: those are fetched in-sandbox (sklearn / OpenML / direct
#: CSV), not pre-staged, so their on-disk shape is up to the generated code.
_DATASET_LAYOUTS: dict[str, str] = {
    "nih_chestxray14": (
        "**NIH ChestX-ray14** — `data/nih_chestxray14/`\n"
        "    - Labels: `Data_Entry_2017.csv`\n"
        "    - Patient splits: `train_val_list.txt`, `test_list.txt`\n"
        "    - Bounding boxes: `BBox_List_2017.csv`\n"
        "    - Images: `images_001/` … `images_012/` (each has an `images/` subdir of PNGs)"
    ),
    "padchest": (
        "**PadChest** — `data/padchest/`\n"
        "    - Labels: `padchest_meta.csv`\n"
        "    - Images: numbered subdirs `0/`, `1/`, `2/`, …"
    ),
}


def load_project_config(projects_root: Path | str, project_id: str) -> dict[str, Any]:
    """Load ``projects/<project_id>/config.toml`` as a dict. ``{}`` if absent.

    Never raises: a malformed or missing config degrades to an empty dict so a
    bad project file can't crash the run loop (the prompt simply renders without
    a context block).
    """
    path = Path(projects_root) / project_id / "config.toml"
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:  # pragma: no cover - defensive
        log.warning(
            "project_context.config_load_failed",
            project_id=project_id, error=str(e), error_type=type(e).__name__,
        )
        return {}


def render_project_context(projects_root: Path | str, project_id: str) -> str:
    """Render the authoritative project-context block for ``project_id``.

    Pulls domain / objective from ``[project]``, the dataset whitelist from
    ``[datasets].allowed``, and the anchor values from ``[experiment.defaults]``.
    Returns ``""`` when no project config is available (the caller then strips
    the placeholder).
    """
    pcfg = load_project_config(projects_root, project_id)
    if not pcfg:
        return ""

    proj = pcfg.get("project", {}) or {}
    domain = proj.get("domain", "unspecified")
    description = str(proj.get("description", "") or "").strip()
    allowed = pcfg.get("datasets", {}).get("allowed", []) or []
    defaults = pcfg.get("experiment", {}).get("defaults", {}) or {}

    lines: list[str] = [
        "## Project context (authoritative)",
        "",
        "The facts below come from THIS project's config and override any example "
        "domain, dataset, or metric mentioned elsewhere in this prompt. Build for "
        "this project only — do not import patterns, datasets, or APIs from another "
        "domain just because the prompt mentions them as illustrations.",
        "",
        f"- **Project**: `{project_id}`",
        f"- **Domain**: {domain}",
    ]
    if description:
        lines.append(f"- **Objective**: {description}")

    lines.append("")
    lines.append("### Datasets")
    if not allowed:
        lines.append(
            "This project uses **NO external datasets**. There is no `data/` "
            "directory and nothing to fetch. Do **not** call `dataset_fetch`/"
            "`dataset_info`, do **not** add data-loading or medical-imaging "
            "scaffolding (no `Dataset` classes, no `patient`/`image` splits, no "
            "`AUROC`), and do **not** invent file paths. All inputs are "
            "generated or analytic — implement them directly in code from the "
            "methodology plan."
        )
    else:
        lines.append(
            "The operator has pre-staged the dataset(s) below. Do **not** call "
            "`dataset_fetch` (re-downloading wastes hours and 50+ GB). Paths are "
            "relative to the sandbox CWD (a `data/` symlink is already set up):"
        )
        for name in allowed:
            layout = _DATASET_LAYOUTS.get(name)
            if layout:
                lines.append(f"- {layout}")
            else:
                lines.append(
                    f"- **{name}** — fetched in-sandbox (sklearn / OpenML / direct "
                    f"CSV); no pre-staged path. The methodology plan names the source."
                )

    if defaults:
        lines.append("")
        lines.append("### Experiment defaults (anchor on these unless the plan overrides)")
        for k, v in defaults.items():
            lines.append(f"- `{k}` = {v}")

    return "\n".join(lines)


def inject_project_context(system_text: str, projects_root: Path | str, project_id: str | None) -> str:
    """Substitute :data:`PLACEHOLDER` in ``system_text`` with the rendered block.

    No-op when the placeholder is absent. When the placeholder is present but no
    context can be rendered (no ``project_id`` or missing config), the placeholder
    is removed so the literal token never reaches the model.
    """
    if PLACEHOLDER not in system_text:
        return system_text
    block = render_project_context(projects_root, project_id) if project_id else ""
    # Collapse the blank line the placeholder usually sits on when block is empty.
    if not block:
        return system_text.replace(PLACEHOLDER + "\n", "").replace(PLACEHOLDER, "")
    return system_text.replace(PLACEHOLDER, block)
