"""Opus-orchestrator mode for code_gen / test_gen.

When the operator picks "Opus orchestrator" for ``code_gen``/``test_gen`` at an
approval gate, the runner (see ``runtime/runner._drive_loop``):

  1. routes that agent's own calls to the manager model (Opus 4.8),
  2. adds the ``delegate`` tool to its toolset, and
  3. appends :data:`ORCHESTRATOR_APPENDIX` to its system prompt.

The manager then plans the work and calls ``delegate`` for each file-level
chunk. Every ``delegate`` call runs a *local* worker agent (``code_worker`` →
qwen2.5-32b, $0) through the normal :func:`runner._invoke_agent` tool loop; the
worker writes the files directly to the sandbox and returns. ``delegate`` then
hands the manager a **compact summary** (sandbox listing + ``check_imports``) so
the manager reads spot-checks, not full files — that's where the cost saving is.

The worker has a write-only toolset (no ``execute``/``handoff``/``delegate``),
so it cannot debug-spin or recurse. Correctness review stays with the manager.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import structlog

from autoscientist.runtime.config import Config, load_config

log = structlog.get_logger("autoscientist.orchestration")

# Override sentinel the console sends (and the runner detects) for orchestrator
# mode. It is NOT a model alias — it selects a *mode* (manager + worker), unlike
# a plain alias override which just swaps the model.
ORCH_OVERRIDE = "orchestrator"

# Only these agents can be orchestrated (they emit the bulk of the code/tests).
ORCHESTRATABLE: frozenset[str] = frozenset({"code_gen", "test_gen"})

# Fallback config values if [orchestrator] is missing/partial in models.toml.
_DEFAULT_MANAGER_MODEL = "claude_opus_48"
_DEFAULT_WORKER_AGENT = "code_worker"
_DEFAULT_WORKER_ROUNDS = 14

# Appended to code_gen.md / test_gen.md when the agent runs as an orchestrator.
ORCHESTRATOR_APPENDIX = """

---

## ORCHESTRATOR MODE (operator-selected for this leg)

You are running as an **orchestrator** on a strong model. You have an extra tool,
`delegate`, that hands a focused, file-level assignment to a fast **local worker**
which writes the file(s) directly into the sandbox. Your job is to **plan and
verify**, not to type out every file yourself — that is what the worker is for.

Workflow:
1. **Decompose** the task into small, well-specified file-level assignments
   (one file, or a couple of tightly-related files, each). Order them so a file
   is only written after the modules it imports from already exist.
2. For each chunk, call
   `delegate(assignment="Write src/foo.py: <exact contract — public names,
   signatures, behaviour, and which existing modules/APIs to match>", files=["src/foo.py"])`.
   Be precise: the worker only sees what you put in `assignment`. Name the exact
   functions/classes/constants it must define and the signatures call-sites expect.
3. `delegate` returns a **compact summary**: the worker's note, the current
   sandbox file listing, and a fresh `check_imports` result. Read it.
4. **You own correctness.** The worker is fast but makes subtle numerical /
   indexing / signature errors. For any file with real logic, call
   `read_sandbox_file` and **verify the math and the API yourself**. Do not trust
   the worker's summary on logic — only on "a file now exists".
5. If something is wrong, either `delegate` a precise fix ("In src/foo.py the
   running-sum in `accumulate()` is off-by-one; rewrite it to …") or make a
   **small** correction yourself with `write_file`. Delegate the bulk; reserve
   `write_file` for small fixes.
6. Repeat until every assigned file exists, `check_imports` returns `ok: true`,
   and you have verified the core logic. **Then hand off exactly as your normal
   contract above describes** (the same `handoff` tool and target).

Keep your own messages short — plan, delegate, read summaries, spot-check, hand
off. The expensive thing is you re-emitting code, so don't: delegate it.
"""


def orchestrator_manager_model(cfg: Config) -> str:
    """Model alias the orchestrator's own calls route through (Opus 4.8)."""
    return cfg.models.get("orchestrator", {}).get("manager_model", _DEFAULT_MANAGER_MODEL)


def orchestrator_worker_agent(cfg: Config) -> str:
    """Agent name the ``delegate`` tool runs for each assignment (local worker)."""
    return cfg.models.get("orchestrator", {}).get("worker_agent", _DEFAULT_WORKER_AGENT)


def _worker_inbound(assignment: str, context: str | None, files: list[str] | None) -> str:
    parts = ["ASSIGNMENT (from the orchestrator — implement exactly this):", assignment.strip()]
    if files:
        parts.append("\nTarget file(s): " + ", ".join(str(f) for f in files))
    if context and context.strip():
        parts.append("\nContext / constraints:\n" + context.strip())
    parts.append(
        "\nWrite the file(s) with write_file, verify with check_imports, then stop "
        "with a one-paragraph plain-text summary. Do NOT hand off."
    )
    return "\n".join(parts)


def delegate_assignment(
    ctx: Any,
    *,
    assignment: str,
    context: str | None = None,
    files: list[str] | None = None,
) -> dict[str, Any]:
    """Run one delegated assignment on the local worker; return a compact summary.

    ``ctx`` is the tool ``ToolContext`` (conn / project_id / projects_root /
    run_id). Reuses the runner's standard agent loop so the worker's calls are
    budget-tracked and recorded as ``code_worker`` messages in the run feed.
    """
    assignment = (assignment or "").strip()
    if not assignment:
        raise ValueError("delegate requires a non-empty 'assignment'")
    if ctx.project_id is None or ctx.projects_root is None or ctx.conn is None:
        raise ValueError("delegate requires conn + project_id + projects_root in context")

    # Imported lazily: orchestration is imported at runner module load, so a
    # top-level `import runner` here would be a circular import.
    from autoscientist.runtime import runner
    from autoscientist.runtime.agent import load_prompt
    from autoscientist.runtime.project_context import inject_project_context
    from autoscientist.tools import check_imports as ci_mod
    from autoscientist.tools import list_sandbox as ls_mod

    cfg = load_config()
    worker_name = orchestrator_worker_agent(cfg)
    worker = runner._build_agent(cfg, worker_name)
    prompt = load_prompt(worker.system_prompt_path)
    prompt = replace(
        prompt,
        system_text=inject_project_context(
            prompt.system_text, ctx.projects_root, ctx.project_id
        ),
    )
    wlog = structlog.get_logger("autoscientist.orchestration").bind(
        run_id=ctx.run_id, worker=worker_name
    )
    max_rounds = runner._agent_max_tool_rounds(cfg, worker_name, _DEFAULT_WORKER_ROUNDS)
    log.info(
        "orchestration.delegate.start",
        worker=worker_name, files=files, assignment_chars=len(assignment),
    )
    result = runner._invoke_agent(
        conn=ctx.conn,
        agent=worker,
        prompt=prompt,
        inbound_text=_worker_inbound(assignment, context, files),
        run_id=ctx.run_id,
        cfg=cfg,
        log=wlog,
        project_id=ctx.project_id,
        max_tool_rounds=max_rounds,
    )

    # Compact, post-assignment view of the sandbox for the manager to spot-check.
    try:
        listing = ls_mod.list_sandbox(
            project_id=ctx.project_id, projects_root=ctx.projects_root, max_entries=80,
        )
        files_now = [e["path"] for e in listing.get("entries", [])]
    except Exception as e:  # never let summarisation crash the delegate call
        files_now = []
        log.warning("orchestration.list_sandbox_failed", error=str(e))
    try:
        ci = ci_mod.check_imports(
            project_id=ctx.project_id, projects_root=ctx.projects_root,
        )
        check = {
            "ok": ci.get("ok"),
            "unresolved": ci.get("unresolved", [])[:20],
            "syntax_errors": ci.get("syntax_errors", [])[:20],
            "summary": ci.get("summary"),
        }
    except Exception as e:
        check = {"ok": None, "error": str(e)}
        log.warning("orchestration.check_imports_failed", error=str(e))

    log.info(
        "orchestration.delegate.done",
        worker=worker_name, n_files=len(files_now),
        imports_ok=check.get("ok"),
    )
    return {
        "worker_model": result.model if result is not None else None,
        "worker_summary": ((result.content if result is not None else "") or "")[:1500],
        "files_in_sandbox": files_now,
        "check_imports": check,
        "note": (
            "The worker wrote files directly to the sandbox. Do NOT trust this "
            "summary on logic — read_sandbox_file any file with real math/indexing "
            "and verify it yourself before relying on it or handing off."
        ),
    }
