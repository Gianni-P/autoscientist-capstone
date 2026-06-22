"""Results validator agent — believability gate before paper writing.

Loops back to code_gen on revise; halts on counterintuitive findings;
advances to figure_gen on pass (figure_gen renders the paper's figures, then
hands to paper_writer). Operator checkpoint #4 reads this output.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="results_validator",
    role="results believability and counterintuitive-finding gate",
    system_prompt_path=Path("results_validator.md"),
    # Forward to figure_gen (it renders figures, then hands to paper_writer);
    # revise back to code_gen. Forward target first (runner._FORWARD_TARGET +
    # _resolve_off_topology rely on forward-first ordering).
    handoff_targets=("figure_gen", "code_gen"),
    tools=("execute",),  # Phase 5 verify/ runs deterministically before this agent
)
