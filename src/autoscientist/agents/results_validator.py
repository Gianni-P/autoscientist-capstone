"""Results validator agent — believability gate before paper writing.

Loops back to code_gen on revise; halts on counterintuitive findings;
advances to paper_writer on pass. Operator checkpoint #4 reads this output.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="results_validator",
    role="results believability and counterintuitive-finding gate",
    system_prompt_path=Path("results_validator.md"),
    handoff_targets=("paper_writer", "code_gen"),
    tools=("execute",),  # Phase 5 verify/ runs deterministically before this agent
)
