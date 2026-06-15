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
    # With only write_file/pdf_parse it can't debug-spin: it writes files and
    # hands off to test_gen, which owns running/testing. dataset_fetch removal
    # also disarms the 50 GB / multi-hour re-download footgun. Restore a tool
    # here only if code_gen genuinely needs it (it doesn't for the current
    # math project).
    tools=("pdf_parse", "write_file"),
)
