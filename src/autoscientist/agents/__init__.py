"""Phase 2 agent registry.

Each agent lives in its own module under ``autoscientist.agents`` and
exports a top-level ``AGENT: Agent`` constant. ``get_agent(name, cfg)``
looks the module up, resolves the relative ``system_prompt_path`` against
``cfg.prompts_dir()``, and returns the fully-resolved ``Agent``.

The runner (``runtime/runner.py``) calls ``get_agent`` with a fallback to
its bare prompt-only ``Agent`` for Phase 1 stub agents (``echo``, ``handoff``)
that don't have their own module. This keeps Phase 1 smoke tests working.

Agents declare ``system_prompt_path`` as a *relative* ``Path`` (just the
filename, e.g. ``Path("lit_review.md")``); ``get_agent`` joins it against
``cfg.prompts_dir()`` so the registry isn't tied to an absolute repo path.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from autoscientist.runtime.agent import Agent

if TYPE_CHECKING:
    from autoscientist.runtime.config import Config

AGENT_NAMES: tuple[str, ...] = (
    "lit_review",
    "idea_gen",
    "idea_critic",
    "methodology",
    "code_gen",
    "test_gen",
    "code_review",
    "results_validator",
    "figure_gen",
    "paper_writer",
    "peer_reviewer",
    "repo_publisher",
    # Not part of the handoff pipeline — invoked only by the `delegate` tool in
    # Opus-orchestrator mode (see agents/code_worker.py, runtime/orchestration.py).
    "code_worker",
)


def get_agent(name: str, cfg: Config) -> Agent | None:
    """Return a fully-resolved Agent for ``name`` or ``None`` if no module exists.

    Resolution order:
      1. Import ``autoscientist.agents.<name>`` and pull its ``AGENT`` const.
      2. Re-bind ``system_prompt_path`` to ``cfg.prompts_dir() / <filename>``
         so the registry is location-independent.
    """
    try:
        mod = importlib.import_module(f"autoscientist.agents.{name}")
    except ModuleNotFoundError:
        return None
    declared = getattr(mod, "AGENT", None)
    if not isinstance(declared, Agent):
        return None
    resolved_path = cfg.prompts_dir() / declared.system_prompt_path.name
    return Agent(
        name=declared.name,
        role=declared.role,
        system_prompt_path=resolved_path,
        handoff_targets=declared.handoff_targets,
        tools=declared.tools,
        mcp_servers=declared.mcp_servers,
    )
