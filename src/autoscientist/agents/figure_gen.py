"""Figure generation agent — turns validated results into paper figures.

Sits between ``results_validator`` and ``paper_writer`` in the pipeline. It
reads the validated experiment results from ``sandbox/runs/``, writes a
matplotlib plotting script (``scripts/generate_figures.py``), RUNS it with
``execute`` to render image files into ``sandbox/figures/``, writes a
``figures/figures.json`` manifest (path + caption + label per figure), then
hands the figure metadata to ``paper_writer``, which embeds each one with
``\\includegraphics``.

Unlike ``code_gen`` (which deliberately has NO ``execute`` to avoid debug-spin),
figure_gen MUST run its plotting code to produce the images — so it is a hybrid:
a code-WRITING agent (write_file/check_imports/handoff, like code_gen) that also
gets ``execute`` (like results_validator) plus ``read_sandbox_file`` /
``list_sandbox`` to find the validated results it plots.

It supports Opus-orchestrator mode (it is in ``orchestration.ORCHESTRATABLE``):
the operator can pick "Opus orchestrator" for it at CP4, in which case Opus
plans + spot-checks while the local ``code_worker`` writes the plot script, and
figure_gen itself runs ``execute`` to render (the worker has no ``execute``).
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="figure_gen",
    role="figure generation from validated results",
    system_prompt_path=Path("figure_gen.md"),
    # Forward to paper_writer; loop back to results_validator if the results on
    # disk are too thin to plot (so the operator sees a real reason at CP4, not
    # a degenerate figure step). Forward target first (see runner._FORWARD_TARGET
    # and _resolve_off_topology, which both rely on forward-first ordering).
    handoff_targets=("paper_writer", "results_validator"),
    # Hybrid toolset: write_file + check_imports + handoff (the code_gen
    # contract) PLUS execute (to render the figures — code_gen omits this on
    # purpose; figure_gen needs it) and read_sandbox_file / list_sandbox (to
    # locate and read the validated result artifacts under runs/).
    tools=(
        "list_sandbox",
        "read_sandbox_file",
        "write_file",
        "execute",
        "check_imports",
        "handoff",
    ),
)
