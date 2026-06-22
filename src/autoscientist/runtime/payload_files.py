"""Safety-net file persister for agent handoff payloads.

Some agents (historically ``code_gen`` on Qwen) emit a
``files: [{path, content}, ...]`` JSON array in their handoff payload
instead of (or in addition to) calling the ``write_file`` tool per file.
The runner uses this module to persist those files to the sandbox
post-hoc, so the next agent sees them on disk.

The primary contract is still the ``write_file`` tool — see
``prompts/code_gen.md``. This module exists so a single regression in
an LLM's structured output doesn't stall the entire pipeline. Every
payload-write is logged with structlog under
``runtime.payload_files`` so operators can see when the fallback fires.

Design notes
~~~~~~~~~~~~

* **Best-effort.** Parse failures, missing keys, sandbox-escape errors,
  and write errors are captured and logged, never raised. The run loop
  must always make progress.
* **Targets a specific shape.** Only persists entries that are dicts
  with both ``path`` (non-empty string) and ``content`` (str-coercible).
  This deliberately excludes ``files_written: [...]`` (the path-only
  summary list that the contract *does* allow in payloads).
* **Idempotent overwrite.** If a file already exists with the same name,
  ``write_file`` overwrites it. That matches the agent's intent — the
  payload represents what the agent *wants* on disk.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

import structlog

from autoscientist.tools import write_file as wf

log = structlog.get_logger("autoscientist.runtime.payload_files")


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Return the first balanced top-level JSON object in ``text`` as a dict.

    Returns ``None`` if no balanced object is present or the object fails
    to parse. Mirrors the logic in ``runtime/runner._maybe_parse_json``
    so callers can rely on the same parser for both file-extraction and
    checkpoint payload preview.
    """
    if not text:
        return None
    # Use a real JSON parser (raw_decode) at each '{' rather than counting
    # braces: a naive depth counter miscounts '}' that appear INSIDE string
    # values (ubiquitous when the payload embeds code/content), truncating the
    # blob so json.loads fails and files are silently never written.
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict):
            return obj
        idx = text.find("{", idx + 1)
    return None


def persist_files_from_payload(
    *,
    payload: str,
    project_id: str,
    projects_root: Path | str,
    agent_name: str,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Persist any ``files: [{path, content}]`` array embedded in ``payload``.

    Returns a list of result dicts (one per attempted entry) so the runner
    can log a single summary line. Each result has ``path``, ``status``
    (``"ok"`` / ``"error"`` / ``"skipped"``), and either ``size_bytes`` or
    ``error``. Never raises.
    """
    parsed = _extract_first_json_object(payload)
    if parsed is None:
        return []
    files = parsed.get("files")
    if not isinstance(files, list) or not files:
        return []

    projects_root = Path(projects_root)
    written: list[dict[str, Any]] = []
    for entry in files:
        if not isinstance(entry, dict):
            written.append({"path": None, "status": "skipped",
                            "reason": "entry not a dict"})
            continue
        path = entry.get("path")
        content = entry.get("content")
        if not isinstance(path, str) or not path:
            written.append({"path": path, "status": "skipped",
                            "reason": "missing or non-string path"})
            continue
        if content is None:
            written.append({"path": path, "status": "skipped",
                            "reason": "missing content"})
            continue
        try:
            res = wf.write_file(
                path=path,
                content=str(content),
                project_id=project_id,
                projects_root=projects_root,
            )
            written.append({
                "path": path,
                "status": "ok",
                "size_bytes": res["size_bytes"],
            })
            log.warning(
                "payload_files.persisted",
                agent=agent_name,
                run_id=run_id,
                project_id=project_id,
                path=path,
                size_bytes=res["size_bytes"],
                note="fallback path - agent should call write_file directly",
            )
        except Exception as e:
            written.append({
                "path": path,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })
            log.error(
                "payload_files.error",
                agent=agent_name,
                run_id=run_id,
                project_id=project_id,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
            )
    return written


# ---------------------------------------------------------------------------
# Reverse direction: rebuild a code_review input payload FROM the sandbox.
# ---------------------------------------------------------------------------

# Skip pathologically large files so one stray artifact can't blow the
# review context; the head is kept with a truncation marker.
_MAX_REVIEW_FILE_BYTES = 100_000


def _collect_py_files(directory: Path, sandbox_root: Path) -> list[dict[str, str]]:
    """Return ``[{path, content}, ...]`` for every ``*.py`` under ``directory``.

    ``path`` is relative to ``sandbox_root`` (e.g. ``src/train.py``) to match
    the ``code_review`` input contract. ``__pycache__`` and unreadable files
    are skipped; over-large files are head-truncated with a marker. Sorted for
    deterministic output. Never raises.
    """
    if not directory.is_dir():
        return []
    out: list[dict[str, str]] = []
    for p in sorted(directory.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            rel = p.relative_to(sandbox_root).as_posix()
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if len(text.encode("utf-8", errors="replace")) > _MAX_REVIEW_FILE_BYTES:
            text = (
                text[:_MAX_REVIEW_FILE_BYTES]
                + f"\n# … truncated by runner (file > {_MAX_REVIEW_FILE_BYTES} bytes) …\n"
            )
        out.append({"path": rel, "content": text})
    return out


def build_code_review_payload_from_sandbox(
    *,
    project_id: str,
    projects_root: Path | str,
) -> str | None:
    """Reconstruct a ``code_review`` input payload from the sandbox on disk.

    ``code_review`` has no file-reading tool (its ``execute`` allowlist is
    pytest/python only), so it depends entirely on receiving the source and
    test file *contents* in its inbound payload. When an upstream agent
    (``code_gen``/``test_gen``) exhausts its tool rounds and hands off with
    empty content, the runner would otherwise feed ``code_review`` the
    ``"(no payload)"`` sentinel; the review then becomes a no-op (it just asks
    for the missing input) and a degenerate CP3 opens carrying that complaint
    (observed 2026-06-12, run_5273a6fe…). This helper rebuilds the
    ``{src_files, test_files, run_cmd_src, run_cmd_tests}`` envelope from the
    sandbox ``src/`` and ``tests/`` trees so the review can proceed against the
    real code/tests.

    Returns the JSON payload string, or ``None`` when the sandbox has no
    reviewable ``.py`` files (the caller keeps its existing fallback).
    Never raises.
    """
    try:
        sandbox = Path(projects_root) / project_id / "sandbox"
        src_files = _collect_py_files(sandbox / "src", sandbox)
        test_files = _collect_py_files(sandbox / "tests", sandbox)
    except Exception as e:  # pragma: no cover - defensive
        log.error(
            "payload_files.review_rebuild_error",
            project_id=project_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    if not src_files and not test_files:
        return None
    payload = {
        "src_files": src_files,
        "test_files": test_files,
        "run_cmd_src": "python src/main.py",
        "run_cmd_tests": "pytest tests/ -x -q",
        "_reconstructed_by_runner": (
            "The upstream agent handed off with empty content; the runner "
            "rebuilt this payload from the sandbox on disk. Review the files "
            "below as usual and emit your verdict + HANDOFF."
        ),
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Rebuild a paper_writer input payload FROM the sandbox results + the plan.
# ---------------------------------------------------------------------------

# Budget on the rebuilt ``results`` object. Without it the helper dumped EVERY
# summary + per-trial .jsonl across every run dir; on a sandbox with many
# accumulated runs this reached ~830 KB / ~208K tokens — larger than the model
# context, so the payload could not be sent (run_fe002213…, 2026-06-19).
#
# All budgets are measured in the SAME serialization the payload ships in
# (``json.dumps(..., indent=2)``, UTF-8 bytes) — measuring compact would
# undercount the wire size by up to ~2x for nested trial rows. The per-file
# ceiling is kept BELOW the aggregate so any single (bounded) artifact always
# fits: the highest-priority file can never blow the budget on its own.
# ``build_paper_writer_payload_from_sandbox`` then applies a final hard ceiling
# on the assembled payload (results + plan + envelope) as a last guarantee.
_MAX_RESULTS_FILE_RAW_BYTES = 200_000  # skip a single artifact whose RAW file exceeds this
_MAX_JSONL_LINES = 60
_PER_FILE_RESULT_BYTES = 120_000       # indent=2 ceiling on one artifact's contribution
_MAX_RESULTS_TOTAL_BYTES = 240_000     # indent=2 ceiling on the whole results object
_MAX_PAYLOAD_BYTES = 360_000           # indent=2 ceiling on the FULL paper_writer payload

# Back-compat alias (referenced elsewhere/tests).
_MAX_RESULTS_FILE_BYTES = _MAX_RESULTS_FILE_RAW_BYTES


def _entry_bytes(rel: str, value: Any) -> int:
    """Bytes this entry adds to the final ``json.dumps(payload, indent=2)``."""
    return len(json.dumps({rel: value}, indent=2).encode("utf-8"))


def _bounded_jsonl_rows(raw: str) -> list[Any]:
    """Parse a ``.jsonl`` into a row list bounded by BOTH line count and bytes.

    Caps at ``_MAX_JSONL_LINES`` rows, then drops trailing rows until the value
    fits ``_PER_FILE_RESULT_BYTES`` (measured indent=2) — so a single fat-row
    trial dump can never exceed the budget on its own (the row cap alone left
    .jsonl uncapped by size). Appends a ``_truncated_by_runner`` marker when
    anything was dropped.
    """
    rows: list[Any] = []
    truncated = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(rows) >= _MAX_JSONL_LINES:
            truncated = True
            break
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Byte-bound (with margin so the appended marker can't push it back over).
    while rows and len(json.dumps(rows, indent=2).encode("utf-8")) > _PER_FILE_RESULT_BYTES - 5_000:
        rows.pop()
        truncated = True
    if truncated and rows:
        rows.append({"_truncated_by_runner":
                     f"rows truncated (cap {_MAX_JSONL_LINES} rows / "
                     f"{_PER_FILE_RESULT_BYTES // 1000} KB)"})
    return rows


def _collect_results_artifacts(
    runs_dir: Path, *, max_total_bytes: int = _MAX_RESULTS_TOTAL_BYTES
) -> dict[str, Any]:
    """Return ``{relpath: parsed-or-text}`` for result files under ``runs_dir``.

    ``*.json`` files are parsed to objects; ``*.jsonl`` files become a bounded
    row list (see :func:`_bounded_jsonl_rows`). ``relpath`` is relative to
    ``runs_dir`` (e.g. ``validator_run/e1_summary.json``). Unreadable /
    over-large / unparseable files are skipped.

    Bounded by ``max_total_bytes`` (indent=2 bytes): ``*_summary.json`` files
    are PRIORITISED, then remaining files smallest-first, each contributing at
    most ``_PER_FILE_RESULT_BYTES``, accumulating until the budget is hit.
    Omitted files (including the lowest-priority overflow) are noted under
    ``_omitted_by_runner``. Summaries are prioritised but not unconditionally
    guaranteed — if summaries alone exceed the budget the smallest still land
    first. Never raises.
    """
    if not runs_dir.is_dir():
        return {}

    candidates: list[tuple[str, Path, str]] = []
    for p in sorted(runs_dir.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        suffix = p.suffix.lower()
        if suffix not in (".json", ".jsonl"):
            continue
        try:
            rel = p.relative_to(runs_dir).as_posix()
        except Exception:
            continue
        candidates.append((rel, p, suffix))

    # Summaries first, then smallest-first, so the high-value aggregate numbers
    # land first and only spare budget is spent on raw per-trial dumps.
    def _priority(item: tuple[str, Path, str]) -> tuple[int, int]:
        _rel, path, _suffix = item
        is_summary = "summary" in path.name.lower()
        try:
            size = path.stat().st_size
        except OSError:
            size = 1 << 62
        return (0 if is_summary else 1, size)

    candidates.sort(key=_priority)

    out: dict[str, Any] = {}
    total = 0
    omitted = 0
    for rel, p, suffix in candidates:
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if suffix == ".json":
            if len(raw.encode("utf-8", errors="replace")) > _MAX_RESULTS_FILE_RAW_BYTES:
                omitted += 1
                continue
            try:
                value: Any = json.loads(raw)
            except json.JSONDecodeError:
                continue
        else:  # .jsonl — bounded by rows AND bytes
            value = _bounded_jsonl_rows(raw)
            if not value:
                continue

        size = _entry_bytes(rel, value)
        # A single artifact larger than the per-file ceiling is dropped outright
        # (this is what makes "the first file always fits" hold). The aggregate
        # check then bounds the running total — applied to EVERY file, including
        # the first.
        if size > _PER_FILE_RESULT_BYTES or total + size > max_total_bytes:
            omitted += 1
            continue
        out[rel] = value
        total += size
    if omitted:
        out["_omitted_by_runner"] = (
            f"{omitted} result file(s) omitted to keep the results object within "
            f"~{max_total_bytes // 1000} KB of model context (summary files are "
            f"prioritised; the largest artifacts are dropped first). If a number "
            f"you need is not present below, omit the claim — never invent one."
        )
    return out


# Reconstruction notes surfaced to the agent when its thin handoff is rebuilt.
_FIGURE_REBUILD_NOTE = (
    "The upstream results_validator handed off a placeholder plan and/or empty "
    "results; the runner rebuilt this payload from the run's methodology plan "
    "and the materialised result JSON under sandbox/runs/. Plot ONLY numbers "
    "that appear verbatim in `results` — never invent data. Write your plotting "
    "script, run it to render the figures into figures/, write the "
    "figures/figures.json manifest, then hand off to paper_writer."
)
_PAPER_REBUILD_NOTE = (
    "The upstream handoff to you was thin (placeholder plan and/or empty "
    "results); the runner rebuilt this payload from the run's methodology plan, "
    "the materialised result JSON under sandbox/runs/, and the figure manifest "
    "under figures/. Every number in your results section MUST come verbatim "
    "from `results`, and every figure you embed MUST be one listed in `figures` "
    "(use its `path` in \\includegraphics). Emit NO unfilled bracket stand-ins "
    "(no result-placeholder, no citation-placeholder) — verify every citation "
    "with citation_check or omit the claim entirely."
)


def _results_envelope(
    *, sandbox: Path, plan_text: str | None, validator_summary: Any, note: str
) -> dict[str, Any] | None:
    """Build the ``{plan, results, validator_summary, _reconstructed_by_runner}``
    envelope from the materialised result JSON under ``sandbox/runs/``.

    Shared by the figure_gen and paper_writer rebuild paths (both ground their
    work in the same validated results + plan). The ``plan`` is embedded as
    structured data when it was itself a JSON-object string. Returns ``None``
    when there are no result artifacts to ground the work. Never raises.
    """
    try:
        results = _collect_results_artifacts(sandbox / "runs")
    except Exception as e:  # pragma: no cover - defensive
        log.error(
            "payload_files.results_envelope_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    if not results:
        return None
    plan: Any = plan_text
    if isinstance(plan_text, str):
        parsed_plan = _extract_first_json_object(plan_text)
        if parsed_plan is not None:
            plan = parsed_plan
    return {
        "plan": plan,
        "results": results,
        "validator_summary": validator_summary,
        "_reconstructed_by_runner": note,
    }


def _collect_figures(sandbox: Path) -> list[dict[str, str]]:
    """Return the figure manifest ``[{path, caption, label}]`` from the sandbox.

    figure_gen writes ``figures/figures.json`` (the authoritative manifest) next
    to the rendered images. Prefer it; if it is missing or unparseable, fall
    back to scanning ``figures/`` for image files and synthesising minimal
    entries (empty caption) so paper_writer still learns the paths. Each ``path``
    is sandbox-relative (e.g. ``figures/fig1.png``) — the path paper_writer
    references with ``\\includegraphics`` and that ``latex_compile`` resolves by
    copying ``figures/`` next to the compiled ``.tex``. Only entries whose image
    actually exists on disk are kept. Never raises.
    """
    figures_dir = sandbox / "figures"
    if not figures_dir.is_dir():
        return []
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    manifest = figures_dir / "figures.json"
    try:
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8", errors="replace"))
            if isinstance(data, dict):
                entries = data.get("figures")
            else:
                entries = data
            for e in entries or []:
                if not isinstance(e, dict):
                    continue
                path = e.get("path")
                if not isinstance(path, str) or not path:
                    continue
                rel = path if path.startswith("figures/") else f"figures/{PurePosixPath(path).name}"
                if rel in seen or not (sandbox / rel).is_file():
                    continue
                seen.add(rel)
                out.append({
                    "path": rel,
                    "caption": str(e.get("caption") or ""),
                    "label": str(e.get("label") or ""),
                })
    except Exception:  # pragma: no cover - defensive; fall through to a disk scan
        out, seen = [], set()
    if out:
        return out
    # Fallback: no usable manifest — scan for rendered image files.
    for p in sorted(figures_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".png", ".pdf", ".jpg", ".jpeg", ".svg"):
            continue
        try:
            rel = p.relative_to(sandbox).as_posix()
        except Exception:
            continue
        if rel in seen:
            continue
        seen.add(rel)
        out.append({"path": rel, "caption": "", "label": ""})
    return out


def build_figure_gen_payload_from_sandbox(
    *,
    project_id: str,
    projects_root: Path | str,
    plan_text: str | None,
    validator_summary: Any = None,
) -> str | None:
    """Reconstruct a ``figure_gen`` input payload from the sandbox + plan.

    figure_gen plots the validated results, so it needs the same authoritative
    ``{plan, results, validator_summary}`` envelope paper_writer used to receive
    directly from results_validator — which (being an LLM) frequently forwards a
    placeholder plan + empty results. This rebuilds that envelope from the run's
    methodology plan and the materialised result JSON under ``sandbox/runs/``.
    Returns the JSON payload string, or ``None`` when there are no result
    artifacts to plot (the caller keeps its existing fallback). Never raises.
    """
    sandbox = Path(projects_root) / project_id / "sandbox"
    env = _results_envelope(
        sandbox=sandbox, plan_text=plan_text,
        validator_summary=validator_summary, note=_FIGURE_REBUILD_NOTE,
    )
    if env is None:
        return None
    return _serialize_within_budget(env)


def build_paper_writer_payload_from_sandbox(
    *,
    project_id: str,
    projects_root: Path | str,
    plan_text: str | None,
    validator_summary: Any = None,
) -> str | None:
    """Reconstruct a ``paper_writer`` input payload from the sandbox + plan.

    ``paper_writer`` drafts the results section from the ``results`` object it
    receives, grounds methods/intro in ``plan``, and embeds the ``figures``
    figure_gen produced. But the upstream agent (an LLM — figure_gen, or
    results_validator before it) does not faithfully echo the large plan +
    materialised result JSON into its handoff payload — observed 2026-06-18
    (run_e93293803c98…): ``"plan": "<the methodology plan>"`` (the literal
    placeholder copied from its prompt) and ``"results": {"metrics": []}``
    (empty). paper_writer then had no numbers to write, so it emitted a shell
    full of ``[RESULT FROM run]`` / ``[CITATION NEEDED]`` placeholders and
    peer_reviewer rejected it on the hard "no unverified citations / no
    unsubstantiated numbers" rules — a degenerate CP5.

    This rebuilds the ``{plan, results, validator_summary, figures}`` envelope
    from the authoritative on-disk sources: the methodology ``plan`` (the run's
    ``code_gen`` input in the DB), the real result artifacts under
    ``sandbox/runs/``, and the figure manifest under ``figures/``. Returns the
    JSON payload string, or ``None`` when there are no result artifacts to ground
    the paper (the caller keeps its existing fallback). Never raises.
    """
    sandbox = Path(projects_root) / project_id / "sandbox"
    env = _results_envelope(
        sandbox=sandbox, plan_text=plan_text,
        validator_summary=validator_summary, note=_PAPER_REBUILD_NOTE,
    )
    if env is None:
        return None
    # Surface the rendered figures so paper_writer can embed them even when the
    # upstream (figure_gen) handoff was thin — the manifest on disk is the
    # authoritative source of figure paths + captions.
    env["figures"] = _collect_figures(sandbox)
    return _serialize_within_budget(env)


def _serialize_within_budget(payload: dict[str, Any]) -> str:
    """Serialize ``payload`` (indent=2) under a hard ceiling on the SENT bytes.

    The per-file/aggregate caps bound ``results`` alone; this is the final
    guarantee on the whole payload (results + plan + envelope), so an oversized
    ``plan`` can't push it past the model context either. Drops the largest
    non-summary result entries until it fits.
    """
    serialized = json.dumps(payload, indent=2)
    results = payload.get("results")
    if not isinstance(results, dict):
        return serialized
    while len(serialized.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        droppable = [k for k in results if k != "_omitted_by_runner" and "summary" not in k.lower()]
        if not droppable:
            droppable = [k for k in results if k != "_omitted_by_runner"]
        if not droppable:
            break
        biggest = max(droppable, key=lambda k: _entry_bytes(k, results[k]))
        del results[biggest]
        results["_omitted_by_runner"] = (
            str(results.get("_omitted_by_runner", "")).rstrip()
            + " (results trimmed further to fit the model context)"
        ).strip()
        serialized = json.dumps(payload, indent=2)
    return serialized


# ---------------------------------------------------------------------------
# Rebuild a peer_reviewer input payload FROM paper_writer's draft + sandbox.
# ---------------------------------------------------------------------------

_MAX_TEX_BYTES = 200_000


def _read_latest_tex(project_dir: Path) -> str | None:
    """Return the most relevant ``.tex`` source under ``project_dir``.

    Prefers a file named ``paper.tex``; otherwise the most-recently-modified
    ``.tex`` under ``latex/`` or ``sandbox/`` (the ``skeleton`` template is
    skipped). Head-truncated past ``_MAX_TEX_BYTES``. Never raises.
    """
    candidates: list[Path] = []
    for sub in ("latex", "sandbox"):
        d = project_dir / sub
        if d.is_dir():
            candidates.extend(
                p for p in d.rglob("*.tex") if "skeleton" not in p.name.lower()
            )
    if not candidates:
        return None
    preferred = [p for p in candidates if p.name.lower() == "paper.tex"] or candidates
    try:
        best = max(preferred, key=lambda p: p.stat().st_mtime)
        text = best.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if len(text.encode("utf-8", errors="replace")) > _MAX_TEX_BYTES:
        text = text[:_MAX_TEX_BYTES] + "\n% … truncated by runner …\n"
    return text or None


def build_peer_reviewer_payload_from_sandbox(
    *,
    project_id: str,
    projects_root: Path | str,
    draft_text: str | None = None,
    plan_text: str | None = None,
    validator_summary: Any = None,
) -> str | None:
    """Reconstruct a ``peer_reviewer`` input payload when paper_writer hands off empty.

    ``peer_reviewer`` reviews the ``{draft, supplementary, context}`` envelope
    paper_writer is supposed to forward. But paper_writer can derail (observed
    2026-06-19, run_fe002213…: it looped calling ``latex_compile`` with no
    ``tex_source``, exhausted its rounds, and emitted an empty final message),
    so peer_reviewer receives ``"(no payload)"`` and degenerates into "please
    send me the draft" — a useless CP5.

    This rebuilds the draft from the most authoritative source available: the
    ``draft_text`` (paper_writer's most recent *substantive* assistant output,
    passed in by the runner from the DB), unwrapped from a ``{"draft"|"paper":
    …}`` envelope when present; else the compiled ``.tex`` on disk. ``plan`` and
    ``validator_summary`` come from the run (passed in by the runner). Returns
    the JSON payload string, or ``None`` when no draft can be recovered (the
    caller keeps its existing fallback). Never raises.
    """
    try:
        project_dir = Path(projects_root) / project_id
        draft: Any = None
        if draft_text and draft_text.strip():
            parsed = _extract_first_json_object(draft_text)
            # Only trust the parsed object if it actually looks like a
            # manuscript. A draft truncated at max_tokens parses to a stray
            # nested object (e.g. a single reference citation), so the
            # first-JSON-object heuristic alone would hand the reviewer a
            # citation instead of the paper (run_fe002213…, 2026-06-19). When
            # it isn't a manuscript, pass the raw draft text — the reviewer can
            # read the (possibly partial) prose.
            # NB: don't include "title" — a reference citation also has a
            # "title", so it would be mistaken for a manuscript. Require a
            # structural manuscript key.
            if isinstance(parsed, dict) and any(
                k in parsed for k in ("sections", "draft", "paper", "abstract")
            ):
                draft = parsed.get("draft") or parsed.get("paper") or parsed.get("sections") or parsed
            else:
                draft = {"text": draft_text}
        if not draft:
            tex = _read_latest_tex(project_dir)
            if tex:
                draft = {"format": "latex", "latex_source": tex}
    except Exception as e:  # pragma: no cover - defensive
        log.error(
            "payload_files.peer_review_rebuild_error",
            project_id=project_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None
    if not draft:
        return None
    plan: Any = plan_text
    if isinstance(plan_text, str):
        parsed_plan = _extract_first_json_object(plan_text)
        if parsed_plan is not None:
            plan = parsed_plan
    payload = {
        "draft": draft,
        "supplementary": None,
        "context": {"plan": plan, "validator_summary": validator_summary},
        "_reconstructed_by_runner": (
            "paper_writer handed off with empty/thin content; the runner rebuilt "
            "this payload from paper_writer's last substantive draft (or the "
            "compiled .tex on disk) plus the run's plan and validator summary. "
            "Review the draft below as usual and emit your structured review + "
            "recommendation + HANDOFF — do NOT ask for the draft to be resent."
        ),
    }
    return json.dumps(payload, indent=2)
