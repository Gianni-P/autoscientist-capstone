"""Code generation agent — implements the next plan step as runnable Python.

Routes to test_gen by default. May route to code_review with a
``BLOCKED:`` note if the plan step is underspecified.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="code_gen",
    role="code synthesis from methodology plan",
    system_prompt_path=Path("code_gen.md"),
    handoff_targets=("test_gen", "code_review"),
    # 2026-06-11: toolset trimmed to match the documented contract in
    # prompts/code_gen.md ("just write the files... Do NOT call execute...
    # Do NOT call dataset_info or dataset_fetch"). qwen3-coder ignored the
    # prose and burned all 30+ tool rounds in an `execute` debug-loop, never
    # emitting HANDOFF -> runner ended the run "completed" with 0 handoffs and
    # an empty console (runs run_0a07…, run_f6fb98…, run_2773…, 2026-06-11).
    # With no `execute` it can't debug-spin: it writes files and hands off to
    # test_gen, which owns running/testing. dataset_fetch removal also disarms
    # the 50 GB / multi-hour re-download footgun.
    #
    # 2026-06-17: added read_sandbox_file + check_imports + handoff (still NO
    # `execute` — the no-debug-spin invariant is preserved). The phantom-import
    # failure (importing names no sibling module defines -> ImportError -> review
    # rejection) is structural: code_gen writes blind. check_imports gives it a
    # read-only AST signal to catch that before handoff; read_sandbox_file lets
    # it re-read its own files; handoff is the reliable, parse-proof alternative
    # to the bare-line `HANDOFF:` directive qwen3-coder routinely failed to emit.
    tools=("pdf_parse", "write_file", "read_sandbox_file", "check_imports", "handoff"),
)
