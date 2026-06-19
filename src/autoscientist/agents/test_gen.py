"""Test generation agent — writes tests targeting plan pitfalls."""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="test_gen",
    role="test synthesis targeting methodology pitfalls",
    system_prompt_path=Path("test_gen.md"),
    handoff_targets=("code_review",),
    # 2026-06-17: added read_sandbox_file + list_sandbox + check_imports + handoff.
    # test_gen keeps `execute` (it runs pytest). check_imports + read_sandbox_file
    # let it align test imports to the source API code_gen actually produced — the
    # test2 run failed partly because tests imported `auroc`/`get_data_loaders` that
    # the source never defined. handoff is the parse-proof routing directive:
    # test_gen repeatedly failed to emit a bare-line `HANDOFF:` and got
    # force-forwarded with an empty payload (run_358912…, 2026-06-12).
    tools=("execute", "write_file", "read_sandbox_file", "list_sandbox", "check_imports", "handoff"),
)
