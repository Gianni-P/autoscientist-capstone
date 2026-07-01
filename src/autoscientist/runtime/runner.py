"""Main agent run loop.

The runner:
  1. opens (or creates) a run row in SQLite
  2. configures structured logging into ``runs/<run_id>/logs/run.jsonl``
  3. invokes the starting agent via the router
  4. parses the agent's output for a ``HANDOFF: <target>`` directive
  5. invokes the next agent with the handoff payload as the user message
  6. repeats until ``DONE`` is returned, ``max_handoffs`` is reached, an
     error fires, OR a checkpoint policy trips (Phase 4: pause and wait
     for the operator)
  7. closes the run row with a terminal status (``completed`` /
     ``failed`` / ``cancelled`` / ``paused``)

Each agent sees only its own message history (system + the inbound
handoff payload as a single user message). State carried across the
chain is exactly the handoff payload — by design (KICKOFF.md §4 #2:
verification > LLM review; small interfaces, deterministic checks).

Phase 4 — checkpoints.
~~~~~~~~~~~~~~~~~~~~~~
After certain agents finish (see ``checkpoints.manager.CHECKPOINT_POLICY``)
the loop opens a pending checkpoint row, marks the run ``paused``, and
returns. The operator resolves the checkpoint via the Streamlit UI and
``resume_run(run_id)`` picks up where the chain stopped, threading the
operator's modifications into the next agent's payload.
"""

from __future__ import annotations

import argparse
import json
import logging as stdlib_logging
import os
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import structlog

from autoscientist.agents import get_agent as registry_get_agent
from autoscientist.checkpoints import manager as checkpoints
from autoscientist.clients.router import route
from autoscientist.runtime import control as run_control
from autoscientist.runtime.agent import Agent, load_prompt
from autoscientist.runtime.config import Config, load_config
from autoscientist.runtime.handoff import DONE, Handoff, parse_handoff
from autoscientist.runtime.orchestration import (
    ORCH_OVERRIDE,
    ORCHESTRATABLE,
    ORCHESTRATOR_APPENDIX,
    orchestrator_manager_model,
)
from autoscientist.runtime.payload_files import (
    build_code_review_payload_from_sandbox,
    build_figure_gen_payload_from_sandbox,
    build_paper_writer_payload_from_sandbox,
    build_peer_reviewer_payload_from_sandbox,
    persist_files_from_payload,
)
from autoscientist.runtime.project_context import inject_project_context
from autoscientist.state.db import (
    end_run,
    open_db,
    record_message,
    start_run,
)
from autoscientist.tools import registry as tool_registry

DEFAULT_MAX_TOOL_ROUNDS = 40
DEFAULT_MAX_CODE_REVIEW_CYCLES = 3

# Verdict-emission safety net for the tool-use loop. A thorough agent
# (observed: code_review on Sonnet) can spend its entire tool-round budget
# investigating and never emit a tool-free final message — so the loop exits
# at the cap with a tool-call result whose content is empty, which then gets
# force-forwarded as an empty payload and opens a degenerate checkpoint
# (run_fbd5651…, 2026-06-18). Two-part fix:
#  (1) _FINALIZE_NUDGE_ROUNDS rounds before the cap, tell the agent to stop
#      calling tools and emit its verdict now;
#  (2) if it still ends on tool calls at the cap, force ONE final completion
#      with tools disabled so a real text verdict is always produced.
_FINALIZE_NUDGE_ROUNDS = 3
_FINALIZE_NUDGE = (
    "SYSTEM: You have {remaining} tool-use round(s) left before this turn is "
    "force-closed. Stop calling tools now and write your FINAL response — your "
    "complete verdict/output as text — ending with a `HANDOFF: <target>` line "
    "(or call the handoff tool). If you keep calling tools you will be cut off "
    "with no output recorded."
)
_FINALIZE_FORCED = (
    "SYSTEM: Tool-use budget exhausted — you may NOT call any more tools. Write "
    "your final response now as plain text: your complete verdict/output, "
    "ending with a `HANDOFF: <target>` line so the pipeline can advance."
)


def _append_user_text(messages: list[dict], provider: str, text: str) -> None:
    """Append an instruction as a user-visible message, provider-shaped.

    Anthropic/mock: fold a text block into the trailing tool_result user
    message when present (two bare user turns in a row is awkward), else add a
    new user message. Ollama/OpenAI: a standalone user message after the tool
    results.
    """
    if provider in ("claude", "mock"):
        tail = messages[-1] if messages else None
        if tail and tail.get("role") == "user" and isinstance(tail.get("content"), list):
            tail["content"].append({"type": "text", "text": text})
        else:
            messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
    else:
        messages.append({"role": "user", "content": text})

# Forward-flow topology for the missing-HANDOFF backstop. When a non-terminal
# agent finishes its turn but the model omits the `HANDOFF: <target>` directive
# (a recurring qwen3-coder failure mode — it emits its JSON payload without the
# directive line, or exhausts its tool rounds mid-task), the runner forces the
# handoff to the agent's FORWARD stage instead of treating the run as terminal
# and ending it "completed" with an empty operator console (see runs
# run_2773…/run_fb9caef…, 2026-06-11). The downstream checkpoint (CP3/CP4/CP5)
# still gates the operator. Keyed by forward *flow*, NOT handoff_targets order —
# code_review's targets are ("code_gen"=revise, "results_validator"=forward), so
# index 0 is not forward. Terminal agents (peer_reviewer → DONE) are absent on
# purpose: a missing directive there stays terminal.
_FORWARD_TARGET: dict[str, str] = {
    # Early linear chain. Without these, an early agent that finishes WITHOUT a
    # `HANDOFF:` line (e.g. a verbose lit_review on a small model that hits its
    # output-token cap mid-summary) was treated as terminal and ended the run
    # before CP1. Forwarding carries the agent's output to the next stage; the
    # relevant checkpoint (CP1/CP2) still gates the operator.
    "lit_review": "idea_gen",
    "idea_gen": "idea_critic",
    "idea_critic": "methodology",
    "methodology": "code_gen",
    "code_gen": "test_gen",
    "test_gen": "code_review",
    "code_review": "results_validator",
    "results_validator": "figure_gen",
    "figure_gen": "paper_writer",
    "paper_writer": "peer_reviewer",
}


def _norm_agent_name(name: str) -> str:
    """Collapse an agent name to alnum-lowercase for tolerant matching.

    Lets an off-topology handoff target that is merely a formatting/typo variant
    of an allowed target (``code-gen`` / ``CodeGen`` / ``code gen``) snap back
    to the real one instead of being treated as a hallucinated destination.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_off_topology(
    from_agent: str, to_agent: str, handoff_targets: tuple[str, ...]
) -> str | None:
    """Correct an off-topology handoff target.

    Returns an allowed target when ``to_agent`` is just a formatting/typo variant
    of one (normalized match); otherwise the agent's forward stage; otherwise
    ``None`` (a terminal agent named a bogus target → end the run). Never returns
    the bogus ``to_agent`` itself, so a hallucinated destination can't route the
    pipeline off its declared rails.
    """
    snapped = next(
        (t for t in handoff_targets
         if _norm_agent_name(t) == _norm_agent_name(to_agent)),
        None,
    )
    return snapped or _FORWARD_TARGET.get(from_agent)

# Below this length, a code_review payload with no file/review structure is
# treated as "thin" (nothing substantive to review) and rebuilt from the
# sandbox. A real structured handoff summary easily clears this; a stray
# conversational fragment does not.
_THIN_REVIEW_PAYLOAD_CHARS = 400

# Structural markers that prove a code_review payload carries real review
# material (file lists, run commands, or a source path) — presence of any one
# means the payload is NOT thin regardless of length.
_REVIEW_PAYLOAD_MARKERS = ("src_files", "files_written", '"files"', "run_cmd", ".py")


def _is_thin_code_review_payload(payload: str) -> bool:
    """True when an inbound code_review payload has nothing real to review.

    code_review has no file-reading tool, so its whole input is the handoff
    payload. The original backstop rebuilt from the sandbox only when that
    payload was empty/whitespace — but a forced handoff can also forward a
    short conversational fragment when an upstream agent runs out of tool
    rounds mid-sentence (observed: test_gen exhausted its rounds and forwarded
    ``"I need to import the configuration constants. Let me fix this:"``).
    Such a payload carries no files and no review structure, so code_review
    hallucinates a "your message got cut off" reply and CP3 opens with that
    garbage instead of a real verdict. Treat a payload that lacks every
    structural marker AND is short as thin so we rebuild from the sandbox.
    """
    if not payload or not payload.strip():
        return True
    lowered = payload.strip().lower()
    if any(marker in lowered for marker in _REVIEW_PAYLOAD_MARKERS):
        return False
    return len(payload.strip()) < _THIN_REVIEW_PAYLOAD_CHARS


# Substrings that prove a paper_writer payload is a placeholder/empty shell
# rather than a real plan + materialised results. ``<the methodology plan>`` is
# the literal placeholder results_validator copies from its own prompt; the
# others are the unfilled markers that signal "no numbers were ever provided".
_PAPER_PLACEHOLDER_MARKERS = (
    "<the methodology plan>",
    "[result from run]",
    "result from run",
)


def _is_thin_paper_writer_payload(payload: str) -> bool:
    """True when an inbound paper_writer payload has no real plan/results.

    paper_writer drafts the results section from ``results`` and grounds the
    paper in ``plan``. results_validator (an LLM) frequently forwards a
    placeholder plan + empty ``results`` (observed 2026-06-18, run_e93293803c98:
    ``"plan": "<the methodology plan>"`` and ``"results": {"metrics": []}``),
    leaving paper_writer nothing to write — it then emits a shell of
    ``[RESULT FROM run]`` / ``[CITATION NEEDED]`` markers that peer_reviewer
    rejects outright. Treat such a payload as thin so the runner rebuilds it
    from the run's plan + the materialised result JSON in the sandbox.

    Thin when: empty/whitespace; OR it carries a placeholder marker; OR it
    parses to JSON whose ``results`` has no usable numeric content (no
    ``terrain_summaries``/``metrics`` rows and no ``mean``-style key).
    """
    if not payload or not payload.strip():
        return True
    lowered = payload.lower()
    if any(marker in lowered for marker in _PAPER_PLACEHOLDER_MARKERS):
        return True
    parsed = _maybe_parse_json(payload)
    if not isinstance(parsed, dict):
        return False  # unparseable but substantial — leave it for the agent
    results = parsed.get("results")
    if results in (None, {}, [], ""):
        return True
    if isinstance(results, dict):
        rows = results.get("terrain_summaries") or results.get("metrics")
        has_rows = isinstance(rows, list) and len(rows) > 0
        # A non-empty results dict with neither rows nor any numeric leaf is a shell.
        has_numbers = any(
            isinstance(v, (int, float)) for v in results.values()
        ) or "mean" in str(results).lower()
        if not has_rows and not has_numbers:
            return True
    return False


def _is_thin_figure_gen_payload(payload: str) -> bool:
    """True when an inbound figure_gen payload has no real results to plot.

    figure_gen draws the paper's figures from the validated ``results``. It is
    now the agent results_validator hands forward to, so it inherits the exact
    failure paper_writer used to have: results_validator (an LLM) frequently
    forwards a placeholder plan + empty results, leaving figure_gen nothing to
    plot. The thinness criterion is identical (no usable result numbers), so we
    reuse :func:`_is_thin_paper_writer_payload`; on a thin payload the runner
    rebuilds figure_gen's input from the run's plan + the result JSON on disk.
    """
    return _is_thin_paper_writer_payload(payload)


def _is_thin_peer_reviewer_payload(payload: str) -> bool:
    """True when an inbound peer_reviewer payload carries no draft to review.

    peer_reviewer reviews the ``{draft, supplementary, context}`` envelope.
    When paper_writer derails (e.g. loops on a failed ``latex_compile`` and
    emits empty content — run_fe002213…, 2026-06-19) the handoff is empty and
    peer_reviewer just asks for the draft, degenerating CP5. Treat as thin so
    the runner rebuilds the input from paper_writer's last real draft + sandbox.

    Thin ONLY when the handoff is essentially empty: empty/whitespace; or a
    short payload whose JSON carries no draft. A *large* payload is reviewable
    as-is and is never thin — even a draft truncated at max_tokens (which can
    parse to a stray nested object like a single citation) is real content the
    reviewer should see, so thinness is never decided from the first JSON
    object when the text is large (run_fe002213…, 2026-06-19).
    """
    if not payload or not payload.strip():
        return True
    if len(payload.strip()) >= _THIN_REVIEW_PAYLOAD_CHARS:
        return False
    parsed = _maybe_parse_json(payload)
    if isinstance(parsed, dict):
        draft = parsed.get("draft") or parsed.get("paper") or parsed.get("sections")
        return draft in (None, {}, [], "")
    return True


# Cumulative USD ceiling for a SINGLE agent invocation's tool-loop. The
# per-call cost_ceiling_usd (router) bounds one call; this bounds the SUM
# across all rounds of one invocation, closing the hole the 2026-05-31 audit
# (item 2) flagged: a loop of individually-sub-ceiling calls that summed to
# $16+. 0 / unset → cap disabled. Overridable per agent in models.toml.
DEFAULT_INVOCATION_CEILING_USD = 10.0


def _max_code_review_cycles(cfg: Config) -> int:
    """Resolve the per-run cap on ``code_review`` firings.

    Order of precedence:
      1. ``AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES`` env var (tests, ad-hoc tuning).
      2. ``config/default.toml`` ``[runtime].max_code_review_cycles``.
      3. ``DEFAULT_MAX_CODE_REVIEW_CYCLES``.

    Invalid (non-int / non-positive) values fall through to the default; we
    log a warning at that point so an operator misconfiguration doesn't
    silently disable the cap.
    """
    raw = os.environ.get("AUTOSCIENTIST_MAX_CODE_REVIEW_CYCLES")
    if raw is None:
        raw = cfg.default.get("runtime", {}).get("max_code_review_cycles")
    if raw is None:
        return DEFAULT_MAX_CODE_REVIEW_CYCLES
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CODE_REVIEW_CYCLES
    return n if n > 0 else DEFAULT_MAX_CODE_REVIEW_CYCLES


def _build_agent(cfg: Config, name: str) -> Agent:
    """Resolve an Agent by name.

    Phase 2: prefer the per-agent module in ``autoscientist.agents.<name>``
    (which carries ``handoff_targets``). Fall back to a bare prompt-only
    Agent so Phase 1 stubs (``echo``, ``handoff``) keep working.
    """
    registered = registry_get_agent(name, cfg)
    if registered is not None:
        if not registered.system_prompt_path.exists():
            raise FileNotFoundError(
                f"system prompt not found for agent '{name}': {registered.system_prompt_path}"
            )
        return registered
    prompt_path = cfg.prompts_dir() / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"system prompt not found for agent '{name}': {prompt_path}"
        )
    return Agent(
        name=name,
        role=name,
        system_prompt_path=prompt_path,
        handoff_targets=(),
        tools=(),
    )


def configure_logging(jsonl_path: Path | None) -> None:
    """Wire structlog through stdlib logging.

    JSON to file (one record per line), Rich console to stdout. Re-entrant:
    clears existing handlers so calling this twice in one process doesn't
    duplicate output.
    """
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    handlers: list[stdlib_logging.Handler] = []
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = stdlib_logging.FileHandler(jsonl_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=pre_chain,
            )
        )
        handlers.append(file_handler)

    console_handler = stdlib_logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False),
            foreign_pre_chain=pre_chain,
        )
    )
    handlers.append(console_handler)

    root = stdlib_logging.getLogger()
    # Close handlers from a previous configure_logging call before replacing
    # them. This runs once per run()/resume_run(); a leaked FileHandler on
    # run.jsonl surfaces as a ResourceWarning, which pyproject's
    # filterwarnings=["error"] turns into spurious test failures. Closing here
    # fixes the leak at its source (StreamHandler.close() leaves stderr open).
    for handler in list(root.handlers):
        try:
            handler.close()
        except Exception:
            pass
    root.handlers = handlers
    root.setLevel(stdlib_logging.INFO)

    structlog.configure(
        processors=[*pre_chain, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _provider_for_agent(cfg: Config, agent_name: str) -> str:
    """Return the model provider configured for the agent ('claude'/'ollama'/'mock')."""
    agents = cfg.models.get("agents", {})
    if agent_name not in agents:
        return "mock"  # tolerant default for stub agents not in models.toml
    model_alias = agents[agent_name].get("model")
    models = cfg.models.get("models", {})
    return models.get(model_alias, {}).get("provider", "mock")


def _agent_max_tool_rounds(cfg: Config, agent_name: str, default: int) -> int:
    """Return per-agent max_tool_rounds from models.toml, or ``default``.

    A non-numeric value in models.toml falls back to ``default`` (with a
    warning) rather than crashing the whole run mid-flight.
    """
    agents = cfg.models.get("agents", {})
    agent_cfg = agents.get(agent_name, {})
    raw = agent_cfg.get("max_tool_rounds", default)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        log = structlog.get_logger("autoscientist.runner")
        log.warning("run.bad_max_tool_rounds", agent=agent_name, value=raw, using=default)
        return default
    return n if n > 0 else default


def _agent_invocation_ceiling(cfg: Config, agent_name: str) -> float:
    """Resolve the cumulative per-invocation USD ceiling for an agent.

    Precedence:
      1. ``AUTOSCIENTIST_INVOCATION_CEILING_USD`` env var (tests / ad-hoc).
      2. ``models.toml [agents.<name>].invocation_ceiling_usd``.
      3. ``models.toml [budget].default_invocation_ceiling_usd``.
      4. :data:`DEFAULT_INVOCATION_CEILING_USD`.

    A resolved value <= 0 disables the cap (treated as "no per-invocation
    limit"); the monthly hard cap in budget.py still applies regardless.
    """
    raw = os.environ.get("AUTOSCIENTIST_INVOCATION_CEILING_USD")
    if raw is not None:
        try:
            return float(raw)
        except ValueError:
            pass  # fall through to config on a malformed override
    agents = cfg.models.get("agents", {})
    agent_cfg = agents.get(agent_name, {})
    budget = cfg.models.get("budget", {})
    for source in (agent_cfg.get("invocation_ceiling_usd"),
                   budget.get("default_invocation_ceiling_usd")):
        if source is None:
            continue
        try:
            return float(source)
        except (TypeError, ValueError):
            log = structlog.get_logger("autoscientist.runner")
            log.warning("run.bad_invocation_ceiling", agent=agent_name, value=source)
            break  # malformed override -> fall through to the safe default
    return DEFAULT_INVOCATION_CEILING_USD


def _invoke_agent(
    *,
    conn,
    agent: Agent,
    prompt,
    inbound_text: str,
    run_id: str,
    cfg: Config,
    log,
    project_id: str,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    model_override: str | None = None,
):
    """Run one agent invocation including its tool-use loop.

    Returns the final assistant ``CompletionResult`` (the one with no tool
    calls; its ``content`` carries any ``HANDOFF: <target>`` directive).

    ``model_override`` (an operator-selected model alias) is forwarded to every
    ``route`` call so the whole invocation runs on the chosen model. ``None``
    uses the agent's configured model.

    Records: one ``user`` message, then for each loop round one ``assistant``
    message and (if there were tool calls) one ``tool`` message per call.
    """
    record_message(
        conn, run_id=run_id, agent_name=agent.name,
        role="user", content=inbound_text,
    )
    conn.commit()

    # Connect any MCP servers this agent needs, registering their tools so the
    # names declared in agent.tools resolve below. Best-effort: a server that
    # can't be reached (no token, Docker down, network down) is logged and
    # skipped; its tools are dropped from this invocation and the agent runs
    # with whatever native tools it has (e.g. repo_publisher still writes the
    # local release tree even when GitHub publishing is unavailable).
    if agent.mcp_servers:
        from autoscientist.tools import mcp_integration

        for server_key in agent.mcp_servers:
            try:
                names = mcp_integration.ensure_server(server_key, cfg)
                log.info("run.mcp_server_ready", agent=agent.name, server=server_key, tools=names)
            except Exception as e:
                log.warning(
                    "run.mcp_server_unavailable",
                    agent=agent.name,
                    server=server_key,
                    error=str(e),
                    error_type=type(e).__name__,
                )

    # Build tool schemas once per invocation. Same set for every round. MCP
    # tools absent from the registry (server unavailable above) are filtered
    # out here so get_specs never raises on a missing dynamic tool.
    effective_tools = [t for t in agent.tools if tool_registry.is_registered(t)]
    effective_tool_set = set(effective_tools)
    dropped_tools = [t for t in agent.tools if t not in effective_tool_set]
    if dropped_tools:
        log.warning("run.tools_unavailable", agent=agent.name, dropped=dropped_tools)
    tool_specs = tool_registry.get_specs(effective_tools) if effective_tools else []
    tools_anth = tool_registry.anthropic_schemas(tool_specs) if tool_specs else None
    tools_oai = tool_registry.openai_schemas(tool_specs) if tool_specs else None
    tools_sig = tool_registry.tools_signature(tool_specs) if tool_specs else None

    provider = _provider_for_agent(cfg, agent.name)
    projects_root = cfg.root / cfg.default.get("paths", {}).get("projects_dir", "projects")

    messages: list[dict] = [{"role": "user", "content": inbound_text}]

    last_result = None
    invocation_ceiling = _agent_invocation_ceiling(cfg, agent.name)
    invocation_cost_usd = 0.0
    for round_idx in range(max_tool_rounds + 1):
        started = time.monotonic()
        result = route(
            conn=conn,
            agent_name=agent.name,
            system=prompt.system_text,
            messages=messages,
            run_id=run_id,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
            tools_anthropic=tools_anth,
            tools_openai=tools_oai,
            tools_signature=tools_sig,
            cfg=cfg,
            project_id=project_id,
            model_override=model_override,
        )
        latency_ms = int((time.monotonic() - started) * 1000)
        last_result = result
        invocation_cost_usd += result.cost_usd or 0.0

        record_message(
            conn, run_id=run_id, agent_name=agent.name,
            role="assistant",
            content=result.content,
            reasoning=result.reasoning,
            model=result.model,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=latency_ms,
        )
        conn.commit()

        log.info(
            "run.agent_done",
            agent=agent.name,
            model=result.model,
            content_chars=len(result.content),
            reasoning_chars=len(result.reasoning) if result.reasoning else 0,
            n_tool_calls=len(result.tool_calls),
            round=round_idx,
        )

        if not result.tool_calls:
            return result

        # Per-invocation budget cap. The call that just returned has already
        # been billed; stop here so we don't dispatch this round's tools and
        # fire yet another (billable) route() call. Mirrors the max-rounds
        # exit: return the last result and let _drive_loop treat it as
        # terminal. Disabled when the ceiling is <= 0.
        if invocation_ceiling > 0 and invocation_cost_usd >= invocation_ceiling:
            log.warning(
                "run.invocation_cost_cap_exceeded",
                agent=agent.name,
                invocation_cost_usd=round(invocation_cost_usd, 4),
                invocation_ceiling_usd=invocation_ceiling,
                round=round_idx,
            )
            return last_result

        # Tool round. Dispatch each call, record outcomes, append round-trip
        # messages for the next route() call.
        ctx = tool_registry.ToolContext(
            conn=conn, project_id=project_id,
            projects_root=projects_root, run_id=run_id,
        )
        dispatch_results: list[tool_registry.DispatchResult] = []
        for tc in result.tool_calls:
            # Gate against the *effective* set (what the model was actually
            # offered), not the static declaration — so a call to an MCP tool
            # whose server was unavailable hits the clean "not allowed" path
            # instead of a confusing unknown_tool error from dispatch.
            allowed = tc.name in effective_tool_set
            if not allowed:
                dr = tool_registry.DispatchResult(
                    name=tc.name, input=tc.input, output=None,
                    error=f"tool not allowed for agent {agent.name}: {tc.name}",
                    duration_ms=0,
                )
                log.warning("run.tool_call_disallowed", agent=agent.name, tool=tc.name)
            elif tc.name == "handoff":
                # Validate the target here so the recorded tool_result reflects
                # acceptance/rejection. The synthesize-and-return (for a valid
                # target) happens after this loop; an invalid target stays an
                # error so the model sees it and can retry within the loop.
                _target = str(tc.input.get("target", "")).strip()
                _allowed_targets = set(agent.handoff_targets) | {DONE}
                if _target in _allowed_targets:
                    dr = tool_registry.DispatchResult(
                        name=tc.name, input=tc.input,
                        output={"target": _target,
                                "summary": str(tc.input.get("summary", "") or ""),
                                "accepted": True},
                        error=None, duration_ms=0,
                    )
                else:
                    dr = tool_registry.DispatchResult(
                        name=tc.name, input=tc.input, output=None,
                        error=(f"invalid handoff target {_target!r}; must be one of "
                               f"{sorted(_allowed_targets)}. Call handoff again with a valid target."),
                        duration_ms=0,
                    )
                    log.warning("run.handoff_tool_invalid_target", agent=agent.name, target=_target)
            else:
                dr = tool_registry.dispatch(tc.name, tc.input, ctx)
            dispatch_results.append(dr)
            record_message(
                conn, run_id=run_id, agent_name=agent.name,
                role="tool",
                content=json.dumps({
                    "tool_use_id": tc.id,
                    "name": dr.name,
                    "input": dr.input,
                    "output": dr.output,
                    "error": dr.error,
                    "duration_ms": dr.duration_ms,
                }, default=str),
                latency_ms=dr.duration_ms,
            )
            conn.commit()

        # Structured handoff: a valid `handoff` tool call ends the turn. Rewrite
        # it into the canonical `HANDOFF: <target>` directive in the content so
        # parse_handoff / payload persistence / checkpoints in _drive_loop are
        # unchanged — this is the parse-proof alternative to the bare-line text
        # directive qwen3-coder routinely failed to emit. An invalid target was
        # turned into a tool error above, so the loop falls through here and the
        # model gets another round to retry.
        _handoff_tc = next((tc for tc in result.tool_calls if tc.name == "handoff"), None)
        if _handoff_tc is not None:
            _target = str(_handoff_tc.input.get("target", "")).strip()
            if _target in (set(agent.handoff_targets) | {DONE}):
                _summary = str(_handoff_tc.input.get("summary", "") or "").strip()
                _prefix = (result.content.rstrip() + "\n\n") if result.content.strip() else ""
                result.content = f"{_prefix}HANDOFF: {_target}" + (f"\n{_summary}" if _summary else "")
                log.info(
                    "run.handoff_tool_used",
                    agent=agent.name, target=_target, round=round_idx,
                )
                return result

        # Build next-round messages. Provider-specific shape.
        if provider in ("claude", "mock"):
            # Anthropic shape: append assistant content blocks then a user
            # message containing tool_result blocks.
            messages.append({
                "role": "assistant",
                "content": result.raw_content_blocks or [
                    {"type": "text", "text": result.content},
                    *[{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                      for tc in result.tool_calls],
                ],
            })
            tool_result_blocks = []
            for tc, dr in zip(result.tool_calls, dispatch_results, strict=True):
                payload_str = json.dumps(
                    {"output": dr.output, "error": dr.error}, default=str
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": payload_str,
                    **({"is_error": True} if dr.error else {}),
                })
            messages.append({"role": "user", "content": tool_result_blocks})
        elif provider == "ollama":
            # OpenAI shape: assistant message includes tool_calls; each
            # tool result is a separate role:tool message keyed by id.
            assistant_msg = result.raw_content_blocks or {
                "role": "assistant",
                "content": result.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                    for tc in result.tool_calls
                ],
            }
            messages.append(assistant_msg)
            for tc, dr in zip(result.tool_calls, dispatch_results, strict=True):
                payload_str = json.dumps(
                    {"output": dr.output, "error": dr.error}, default=str
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": payload_str,
                })
        else:
            log.error("run.tool_round_unsupported_provider", provider=provider)
            return result

        # (1a) Nearing the round cap: nudge the agent to stop investigating and
        # emit its verdict, so it returns a real tool-free message before we run
        # out of rounds. Fires once (only one round_idx makes remaining match).
        remaining = max_tool_rounds - round_idx
        if remaining == _FINALIZE_NUDGE_ROUNDS:
            _append_user_text(messages, provider, _FINALIZE_NUDGE.format(remaining=remaining))
            log.info("run.finalize_nudge_injected", agent=agent.name, remaining=remaining)

    log.warning(
        "run.tool_loop_max_rounds_reached", agent=agent.name, rounds=max_tool_rounds,
    )
    # (1b) The agent burned every round on tool calls and never emitted a
    # tool-free verdict, so last_result.content is empty/partial. Force ONE
    # final completion with tools disabled — guarantees a real text verdict
    # instead of force-forwarding an empty payload (degenerate checkpoint).
    if last_result is not None and last_result.tool_calls:
        _append_user_text(messages, provider, _FINALIZE_FORCED)
        forced = route(
            conn=conn,
            agent_name=agent.name,
            system=prompt.system_text,
            messages=messages,
            run_id=run_id,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
            tools_anthropic=None,
            tools_openai=None,
            tools_signature=None,
            cfg=cfg,
            project_id=project_id,
            model_override=model_override,
        )
        record_message(
            conn, run_id=run_id, agent_name=agent.name,
            role="assistant",
            content=forced.content,
            reasoning=forced.reasoning,
            model=forced.model,
            prompt_tokens=forced.prompt_tokens,
            completion_tokens=forced.completion_tokens,
        )
        conn.commit()
        log.warning(
            "run.verdict_forced_after_max_rounds",
            agent=agent.name,
            content_chars=len(forced.content),
        )
        return forced
    return last_result


def _maybe_parse_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of ``text``. Best-effort.

    The checkpoint envelope stores both the raw assistant text and a
    parsed dict so the Streamlit UI can render structured payloads
    without re-parsing on every page load. Failures are swallowed —
    a checkpoint with ``parsed=None`` falls back to raw display.
    """
    if not text:
        return None
    # Use a real JSON parser (raw_decode) at each '{' rather than counting
    # braces: a naive depth counter miscounts '}' inside string values (common
    # when the assistant text embeds code), truncating the blob so it fails to
    # parse and the checkpoint preview / thin-payload detection silently no-op.
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


def _fetch_plan_and_validator(conn, run_id: str) -> tuple[str | None, object]:
    """Return ``(plan_text, validator_summary)`` for a run from the DB.

    Authoritative sources for paper_writer payload reconstruction:

    * **plan** — the ``code_gen`` agent's first inbound (``role='user'``)
      message, which carries the full CP2-approved methodology plan verbatim
      (results_validator's forwarded ``plan`` is an unreliable placeholder).
    * **validator_summary** — results_validator's *most recent* non-empty
      assistant message, parsed to a dict when possible. Recency (not length)
      is what matters: results_validator can fire several times across CP3/CP4
      revise cycles, and the operative verdict is the last one — the one that
      drove the forward handoff to paper_writer (an earlier, longer ``revise``
      message is stale).

    Best-effort: any missing piece returns ``None`` for that slot. Never raises.
    """
    plan_text: str | None = None
    validator_summary: object = None
    try:
        row = conn.execute(
            "SELECT content FROM messages WHERE run_id=? AND agent_name='code_gen' "
            "AND role='user' ORDER BY created_at ASC, rowid ASC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is not None:
            plan_text = row[0]
    except Exception:  # pragma: no cover - defensive
        plan_text = None
    try:
        row = conn.execute(
            "SELECT content FROM messages WHERE run_id=? AND agent_name='results_validator' "
            "AND role='assistant' AND TRIM(content) != '' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        if row is not None and row[0]:
            validator_summary = _maybe_parse_json(row[0]) or row[0]
    except Exception:  # pragma: no cover - defensive
        validator_summary = None
    return plan_text, validator_summary


def _fetch_paper_draft(conn, run_id: str) -> str | None:
    """paper_writer's most recent *non-empty* assistant message (the draft).

    Used to rebuild peer_reviewer's input when paper_writer hands off empty: an
    earlier paper_writer turn produced a real draft even if the final one was
    blank, so the most recent substantive output is the right thing to review.
    Never raises.
    """
    try:
        row = conn.execute(
            "SELECT content FROM messages WHERE run_id=? AND agent_name='paper_writer' "
            "AND role='assistant' AND TRIM(content) != '' "
            "ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return row[0] if row is not None and row[0] else None
    except Exception:  # pragma: no cover - defensive
        return None


def _drive_loop(
    *,
    conn,
    cfg: Config,
    log,
    run_id: str,
    project_id: str,
    starting_agent: str,
    starting_payload: str,
    max_handoffs: int,
    max_tool_rounds: int,
    enable_checkpoints: bool,
    model_overrides: dict[str, str] | None = None,
    handoffs_so_far: int = 0,
    code_review_cycles_so_far: int = 0,
) -> tuple[str, str | None, int]:
    """Run the chain forward. Returns (final_status, note, handoffs_used).

    Halts on:
      * terminal HANDOFF: DONE          → ``completed``
      * no HANDOFF directive in output  → ``completed`` (treated as terminal)
      * checkpoint policy trips         → ``paused``
      * max_handoffs reached            → ``failed``
      * KeyboardInterrupt               → ``cancelled``
      * any other exception             → ``failed``

    The loop body is shared by :func:`run` (fresh start) and
    :func:`resume_run` (post-checkpoint continuation).
    """
    current_agent_name = starting_agent
    current_payload = starting_payload
    handoffs = handoffs_so_far
    final_status = "completed"
    note: str | None = None
    # Operator-selected per-leg model overrides ({agent_name: model_alias}, or
    # the ORCH_OVERRIDE sentinel). Lives only for this _drive_loop pass — the
    # next checkpoint approval supplies a fresh map (see resume_run).
    model_overrides = model_overrides or {}

    # Cap on consecutive code_review firings in this _drive_loop pass. A
    # CHECKPOINT resume passes 0 (operator approval at CP3 intentionally resets
    # the cap); a MANUAL-pause resume threads the saved count back so a pause
    # mid-revise-loop doesn't silently hand the model a fresh revision budget.
    code_review_cycles = code_review_cycles_so_far
    max_code_review_cycles = _max_code_review_cycles(cfg)

    try:
        while True:
            agent = _build_agent(cfg, current_agent_name)
            prompt = load_prompt(agent.system_prompt_path)
            # Substitute the {{PROJECT_CONTEXT}} marker with this project's own
            # domain / objective / dataset facts (from projects/<id>/config.toml)
            # so the shared prompt is not hardcoded to one domain. Removing the
            # baked-in chest-xray block stopped the local model from emitting
            # medical scaffolding into unrelated projects (see project_context).
            # No-op for prompts without the marker.
            _projects_root = cfg.root / cfg.default.get("paths", {}).get("projects_dir", "projects")
            prompt = replace(
                prompt,
                system_text=inject_project_context(prompt.system_text, _projects_root, project_id),
            )

            # Per-leg model override (operator-selected at the approval gate;
            # applies only to this _drive_loop pass). A plain alias swaps the
            # model for this agent; the ORCH_OVERRIDE sentinel on an
            # orchestratable agent (code_gen/test_gen) switches it to
            # Opus-orchestrator mode — route to the manager model, add the
            # `delegate` tool, and append the orchestrator playbook to the prompt.
            override_alias = model_overrides.get(current_agent_name)
            route_override: str | None = None
            if override_alias == ORCH_OVERRIDE and current_agent_name in ORCHESTRATABLE:
                route_override = orchestrator_manager_model(cfg)
                agent = replace(
                    agent, tools=tuple(dict.fromkeys((*agent.tools, "delegate")))
                )
                prompt = replace(prompt, system_text=prompt.system_text + ORCHESTRATOR_APPENDIX)
                log.info(
                    "run.orchestrator_mode",
                    agent=current_agent_name, manager_model=route_override,
                )
            elif override_alias and override_alias != ORCH_OVERRIDE:
                route_override = override_alias
                log.info("run.model_override", agent=current_agent_name, model=route_override)

            inbound_text = current_payload or "(no payload)"

            # code_review has no file-reading tool, so a handoff payload with
            # nothing real to review — empty (upstream agent exhausted its tool
            # rounds and emitted no content) OR a short structureless fragment
            # (a forced handoff forwarding a mid-sentence scrap) — would make
            # the review a no-op: it opens a degenerate CP3 carrying a
            # "(no payload)"/"your message got cut off" complaint instead of a
            # verdict. Rebuild its input from the sandbox on disk so the review
            # always runs against the real code/tests. Single guard here covers
            # every path that can feed code_review a thin payload (forced
            # handoff, resume, manual re-entry).
            if current_agent_name == "code_review" and _is_thin_code_review_payload(
                current_payload or ""
            ):
                projects_root = cfg.root / cfg.default.get("paths", {}).get(
                    "projects_dir", "projects"
                )
                rebuilt = build_code_review_payload_from_sandbox(
                    project_id=project_id, projects_root=projects_root,
                )
                if rebuilt:
                    log.warning(
                        "run.code_review_payload_reconstructed",
                        agent=current_agent_name,
                        chars=len(rebuilt),
                        thin_payload_chars=len((current_payload or "").strip()),
                    )
                    inbound_text = rebuilt

            # figure_gen plots the validated results, so — like paper_writer — a
            # placeholder plan + empty results from results_validator leaves it
            # nothing to draw. Rebuild its input from the run's real plan (the
            # code_gen input in the DB) and the materialised result JSON in the
            # sandbox so the figures come from actual numbers. The figures
            # themselves are produced downstream by figure_gen's own execute
            # call; this only ensures it RECEIVES the results to plot.
            if current_agent_name == "figure_gen" and _is_thin_figure_gen_payload(
                current_payload or ""
            ):
                projects_root = cfg.root / cfg.default.get("paths", {}).get(
                    "projects_dir", "projects"
                )
                plan_text, validator_summary = _fetch_plan_and_validator(conn, run_id)
                rebuilt = build_figure_gen_payload_from_sandbox(
                    project_id=project_id,
                    projects_root=projects_root,
                    plan_text=plan_text,
                    validator_summary=validator_summary,
                )
                if rebuilt:
                    log.warning(
                        "run.figure_gen_payload_reconstructed",
                        agent=current_agent_name,
                        chars=len(rebuilt),
                        thin_payload_chars=len((current_payload or "").strip()),
                        had_plan=bool(plan_text),
                        had_validator_summary=validator_summary is not None,
                    )
                    inbound_text = rebuilt

            # paper_writer drafts the results section from its inbound `results`
            # and grounds the paper in `plan`. results_validator (an LLM) often
            # forwards a placeholder plan + empty results (observed 2026-06-18,
            # run_e93293803c98), leaving paper_writer nothing to write — it then
            # emits a shell of [RESULT FROM run]/[CITATION NEEDED] markers that
            # peer_reviewer rejects outright (a degenerate CP5). Rebuild its
            # input from the run's real plan (the code_gen input in the DB) and
            # the materialised result JSON in the sandbox so the paper is
            # written against actual numbers.
            if current_agent_name == "paper_writer" and _is_thin_paper_writer_payload(
                current_payload or ""
            ):
                projects_root = cfg.root / cfg.default.get("paths", {}).get(
                    "projects_dir", "projects"
                )
                plan_text, validator_summary = _fetch_plan_and_validator(conn, run_id)
                rebuilt = build_paper_writer_payload_from_sandbox(
                    project_id=project_id,
                    projects_root=projects_root,
                    plan_text=plan_text,
                    validator_summary=validator_summary,
                )
                if rebuilt:
                    log.warning(
                        "run.paper_writer_payload_reconstructed",
                        agent=current_agent_name,
                        chars=len(rebuilt),
                        thin_payload_chars=len((current_payload or "").strip()),
                        had_plan=bool(plan_text),
                        had_validator_summary=validator_summary is not None,
                    )
                    inbound_text = rebuilt

            # peer_reviewer reviews paper_writer's {draft, ...} envelope. When
            # paper_writer derails (e.g. loops on a failed latex_compile and
            # emits empty content — run_fe002213…, 2026-06-19) the handoff is
            # empty and peer_reviewer just asks for the draft → a degenerate CP5.
            # Rebuild its input from paper_writer's last real draft (DB) plus the
            # run's plan + validator summary so the review always runs.
            if current_agent_name == "peer_reviewer" and _is_thin_peer_reviewer_payload(
                current_payload or ""
            ):
                projects_root = cfg.root / cfg.default.get("paths", {}).get(
                    "projects_dir", "projects"
                )
                draft_text = _fetch_paper_draft(conn, run_id)
                plan_text, validator_summary = _fetch_plan_and_validator(conn, run_id)
                rebuilt = build_peer_reviewer_payload_from_sandbox(
                    project_id=project_id,
                    projects_root=projects_root,
                    draft_text=draft_text,
                    plan_text=plan_text,
                    validator_summary=validator_summary,
                )
                if rebuilt:
                    log.warning(
                        "run.peer_reviewer_payload_reconstructed",
                        agent=current_agent_name,
                        chars=len(rebuilt),
                        thin_payload_chars=len((current_payload or "").strip()),
                        had_draft=bool(draft_text),
                    )
                    inbound_text = rebuilt

            result = _invoke_agent(
                conn=conn,
                agent=agent,
                prompt=prompt,
                inbound_text=inbound_text,
                run_id=run_id,
                cfg=cfg,
                log=log,
                project_id=project_id,
                max_tool_rounds=_agent_max_tool_rounds(cfg, current_agent_name, max_tool_rounds),
                model_override=route_override,
            )

            if current_agent_name == "code_review":
                code_review_cycles += 1

            # Regression detector: paper_writer must never ship unfilled
            # placeholders. Even after the input is reconstructed above, a
            # weaker model could still leave a [RESULT FROM run]/[CITATION
            # NEEDED]/CITATION_NEEDED_* marker — which is a guaranteed
            # peer_reviewer rejection. Surface it loudly in the run log (and so
            # at CP5) instead of letting it look like a clean draft.
            if current_agent_name == "paper_writer":
                _lc = (result.content or "").lower()
                leftover = [
                    m for m in ("[result from run]", "result from run",
                                "[citation needed]", "citation_needed")
                    if m in _lc
                ]
                if leftover:
                    log.warning(
                        "run.paper_writer_unfilled_placeholders",
                        agent=current_agent_name,
                        markers=leftover,
                        note="draft still carries placeholders — peer_reviewer "
                             "will reject; check that results were materialised.",
                    )

            # Safety net for the "agent emits files: [{path, content}, ...] in
            # JSON instead of calling write_file per file" failure mode (see
            # runtime/payload_files.py). Called on the raw assistant content
            # so the fallback fires regardless of where the JSON sits relative
            # to the HANDOFF directive. No-op when there are no embedded files.
            projects_root_for_files = cfg.root / cfg.default.get("paths", {}).get("projects_dir", "projects")
            payload_writes = persist_files_from_payload(
                payload=result.content,
                project_id=project_id,
                projects_root=projects_root_for_files,
                agent_name=current_agent_name,
                run_id=run_id,
            )
            if payload_writes:
                n_ok = sum(1 for w in payload_writes if w.get("status") == "ok")
                n_err = sum(1 for w in payload_writes if w.get("status") == "error")
                n_skipped = sum(1 for w in payload_writes if w.get("status") == "skipped")
                log.warning(
                    "run.payload_files_persisted",
                    agent=current_agent_name,
                    n_ok=n_ok,
                    n_error=n_err,
                    n_skipped=n_skipped,
                    paths=[w.get("path") for w in payload_writes if w.get("status") == "ok"],
                )

            handoff = parse_handoff(result.content, from_agent=current_agent_name)
            if handoff is None:
                forward_target = _FORWARD_TARGET.get(current_agent_name)
                if forward_target is None:
                    # Genuinely terminal agent (e.g. peer_reviewer) emitted no
                    # directive — treat as the clean end of the run.
                    log.info("run.no_handoff_terminal", agent=current_agent_name)
                    break
                # Non-terminal agent finished but the model omitted the
                # `HANDOFF: <target>` line (or ran out of tool rounds mid-task).
                # Force the forward handoff so the pipeline advances and the next
                # checkpoint still gates the operator, rather than silently
                # ending "completed" with an empty console. The agent's full
                # output becomes the forwarded payload.
                log.warning(
                    "run.forced_handoff_missing_directive",
                    agent=current_agent_name,
                    forced_to=forward_target,
                    content_chars=len(result.content),
                )
                handoff = Handoff(
                    from_agent=current_agent_name,
                    to_agent=forward_target,
                    payload=result.content,
                )
            if handoff.is_terminal:
                log.info("run.handoff_done", agent=current_agent_name)
                # Even on terminal handoff, the just-finished agent might be
                # gated by a checkpoint (e.g. peer_reviewer → DONE still wants
                # operator approval at draft review).
                stage_info = (
                    checkpoints.stage_for_agent(current_agent_name, handoff_to=DONE)
                    if enable_checkpoints else None
                )
                if stage_info is not None:
                    stage, _stage_name = stage_info
                    cp_id = checkpoints.open_checkpoint(
                        conn,
                        run_id=run_id,
                        stage=stage,
                        from_agent=current_agent_name,
                        to_agent=DONE,
                        agent_output_raw=result.content,
                        default_payload="",
                        parsed=_maybe_parse_json(result.content),
                    )
                    log.info(
                        "run.checkpoint_opened",
                        agent=current_agent_name,
                        stage=stage,
                        checkpoint_id=cp_id,
                        terminal=True,
                    )
                    final_status = "paused"
                    note = f"awaiting operator at stage {stage} ({cp_id})"
                break

            if (
                agent.handoff_targets
                and handoff.to_agent != DONE
                and handoff.to_agent not in agent.handoff_targets
            ):
                # The model named a successor this agent isn't allowed to hand
                # to. Don't blindly follow a hallucinated target (it can skip
                # stages or jump anywhere in the pipeline): snap to an allowed
                # variant, else redirect to the forward stage, else (terminal
                # agent) end the run cleanly.
                corrected = _resolve_off_topology(
                    current_agent_name, handoff.to_agent, agent.handoff_targets
                )
                log.warning(
                    "run.handoff_off_topology",
                    from_agent=current_agent_name,
                    to_agent=handoff.to_agent,
                    allowed=list(agent.handoff_targets),
                    corrected_to=corrected,
                )
                if corrected is None:
                    log.info("run.off_topology_terminal", agent=current_agent_name)
                    break
                handoff = replace(handoff, to_agent=corrected)

            stage_info = (
                checkpoints.stage_for_agent(current_agent_name, handoff_to=handoff.to_agent)
                if enable_checkpoints else None
            )

            # Loop-cap forced CP3: when code_review wants another revise but
            # has already fired the configured cap of times in this _drive_loop
            # session, escalate to the operator via a CP3 carrying
            # extra.loop_cap_exceeded=true. stage_for_agent has returned None
            # because the policy excludes revise transitions; this branch
            # re-introduces the gate to bound runaway revision loops.
            loop_cap_extra: dict[str, object] | None = None
            if (
                enable_checkpoints
                and stage_info is None
                and current_agent_name == "code_review"
                and handoff.to_agent == "code_gen"
                and code_review_cycles >= max_code_review_cycles
            ):
                stage_info = (3, "preliminary_review")
                loop_cap_extra = {
                    "loop_cap_exceeded": True,
                    "cycles": code_review_cycles,
                    "max_cycles": max_code_review_cycles,
                }
                log.warning(
                    "run.code_review_loop_cap_exceeded",
                    cycles=code_review_cycles,
                    max_cycles=max_code_review_cycles,
                )

            if stage_info is not None:
                stage, _stage_name = stage_info
                cp_id = checkpoints.open_checkpoint(
                    conn,
                    run_id=run_id,
                    stage=stage,
                    from_agent=current_agent_name,
                    to_agent=handoff.to_agent,
                    agent_output_raw=result.content,
                    default_payload=handoff.payload,
                    parsed=_maybe_parse_json(result.content),
                    extra=loop_cap_extra,
                )
                log.info(
                    "run.checkpoint_opened",
                    agent=current_agent_name,
                    stage=stage,
                    checkpoint_id=cp_id,
                    next_agent=handoff.to_agent,
                    loop_cap_exceeded=bool(loop_cap_extra),
                )
                final_status = "paused"
                note = f"awaiting operator at stage {stage} ({cp_id})"
                break

            # Manual pause poll: between agents, after a clean handoff that
            # *didn't* trip a checkpoint. Always honoured (independent of
            # ``enable_checkpoints``, which gates the KICKOFF §7 policy) so
            # an operator who clicked Pause never gets ignored. The saved
            # state lets ``resume_run`` rebuild the loop counters.
            if run_control.is_pause_requested(conn, run_id):
                run_control.save_pause_state(
                    conn,
                    run_id=run_id,
                    next_agent=handoff.to_agent,
                    next_payload=handoff.payload,
                    handoffs_so_far=handoffs,
                    code_review_cycles=code_review_cycles,
                )
                conn.commit()
                log.info(
                    "run.manual_pause",
                    next_agent=handoff.to_agent,
                    handoffs_done=handoffs,
                    code_review_cycles=code_review_cycles,
                )
                final_status = "paused"
                note = "manual_pause"
                break

            handoffs += 1
            if handoffs >= max_handoffs:
                final_status = "failed"
                note = f"max_handoffs ({max_handoffs}) reached"
                log.warning("run.max_handoffs_reached", handoffs=handoffs)
                break

            current_agent_name = handoff.to_agent
            current_payload = handoff.payload

    except KeyboardInterrupt:
        final_status = "cancelled"
        note = "operator interrupt"
        log.warning("run.cancelled")
    except Exception as e:
        final_status = "failed"
        note = f"{type(e).__name__}: {e}"
        log.exception("run.failed", error_type=type(e).__name__)

    return final_status, note, handoffs


def run(
    *,
    starting_agent: str,
    project_id: str,
    initial_payload: str = "",
    max_handoffs: int = 50,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    enable_checkpoints: bool = True,
    model_overrides: dict[str, str] | None = None,
    cfg: Config | None = None,
) -> str:
    """Drive the agent run loop from a fresh start. Returns the run_id.

    ``enable_checkpoints`` defaults to True per KICKOFF.md §4 #5 (no
    autonomous run-to-completion). The Phase 1–3.5 substrate smoke
    tests opt out so they exercise the chain end-to-end without the
    new gate.
    """
    cfg = cfg or load_config()
    db_path = cfg.db_path()

    conn = open_db(db_path)
    try:
        run_id = start_run(
            conn,
            project_id=project_id,
            config_snapshot={
                "starting_agent": starting_agent,
                "max_handoffs": max_handoffs,
                "enable_checkpoints": enable_checkpoints,
            },
        )
        conn.commit()

        log_path = cfg.runs_dir() / run_id / "logs" / "run.jsonl"
        configure_logging(log_path)
        log = structlog.get_logger("autoscientist.runner").bind(run_id=run_id)
        log.info(
            "run.start",
            project_id=project_id,
            starting_agent=starting_agent,
            db_path=str(db_path),
            log_path=str(log_path),
            enable_checkpoints=enable_checkpoints,
        )

        final_status, note, handoffs = _drive_loop(
            conn=conn,
            cfg=cfg,
            log=log,
            run_id=run_id,
            project_id=project_id,
            starting_agent=starting_agent,
            starting_payload=initial_payload,
            max_handoffs=max_handoffs,
            max_tool_rounds=max_tool_rounds,
            enable_checkpoints=enable_checkpoints,
            model_overrides=model_overrides,
        )

        end_run(conn, run_id, final_status, note)
        conn.commit()
        log.info("run.end", status=final_status, handoffs=handoffs, note=note)
        return run_id
    finally:
        conn.close()


_NUDGE_RE = re.compile(r"\n\nOPERATOR_NUDGE:.*\Z", re.S)


def _latest_inbound_for_agent(
    conn, run_id: str, agent_name: str
) -> str | None:
    """The most recent ``role='user'`` payload delivered to ``agent_name``.

    This is the exact text the agent last ran on (``_invoke_agent`` records it
    before the tool loop), so replaying it re-runs the agent faithfully. Used
    by the operator "re-run with nudge" path.
    """
    row = conn.execute(
        "SELECT content FROM messages WHERE run_id = ? AND agent_name = ? "
        "AND role = 'user' ORDER BY rowid DESC LIMIT 1",
        (run_id, agent_name),
    ).fetchone()
    return row["content"] if row else None


def _apply_nudge(inbound: str | None, nudge: str | None) -> str:
    """Append an ``OPERATOR_NUDGE`` block, replacing any prior one.

    Repeated re-runs replace (not stack) the nudge so the inbound doesn't grow
    unbounded across successive operator re-runs.
    """
    base = _NUDGE_RE.sub("", inbound or "")
    nudge = (nudge or "").strip()
    return f"{base}\n\nOPERATOR_NUDGE: {nudge}" if nudge else base


def _model_overrides_from_op(op: dict | None) -> dict[str, str]:
    """Pull the operator's per-leg model overrides out of a checkpoint's
    ``operator_input``. Tolerant: non-dict / empty values are dropped so a
    malformed payload never breaks the resume."""
    mo = (op or {}).get("model_overrides")
    if not isinstance(mo, dict):
        return {}
    return {str(k): str(v) for k, v in mo.items() if v}


def resume_run(
    run_id: str,
    *,
    max_handoffs: int = 50,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    cfg: Config | None = None,
) -> str:
    """Resume a paused run.

    Two paths:

    * **Manual-pause resume.** If ``run_controls`` has saved state for this
      run (``paused_at`` set, ``next_agent`` known), the runner picks up
      from the saved agent + payload + handoff counter. No checkpoint
      interaction. The row is cleared after the state is read.
    * **Checkpoint-based resume** (the original Phase 4 path). If no
      manual-pause state exists, the latest checkpoint must be resolved
      (``approved`` / ``modified``) and the operator's decision feeds the
      next agent's payload.

    Raises if the run isn't in ``paused`` status, or if a manual-pause
    state isn't present and the latest checkpoint is still ``pending``.
    """
    cfg = cfg or load_config()
    db_path = cfg.db_path()

    conn = open_db(db_path)
    try:
        row = conn.execute(
            "SELECT run_id, project_id, status FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown run: {run_id}")
        if row["status"] != "paused":
            raise RuntimeError(
                f"run {run_id} is not paused (status={row['status']})"
            )
        project_id = row["project_id"]

        log_path = cfg.runs_dir() / run_id / "logs" / "run.jsonl"
        configure_logging(log_path)
        log = structlog.get_logger("autoscientist.runner").bind(run_id=run_id)

        # --- Path 1: manual-pause resume -----------------------------------
        pause = run_control.read_pause_state(conn, run_id)
        if pause is not None and pause.is_active:
            next_agent = pause.next_agent
            next_payload = pause.next_payload or ""
            handoffs_so_far = pause.handoffs_so_far or 0
            code_review_cycles_so_far = pause.code_review_cycles or 0
            log.info(
                "run.resume_from_manual_pause",
                next_agent=next_agent,
                handoffs_so_far=handoffs_so_far,
                paused_at=pause.paused_at,
            )
            record_message(
                conn,
                run_id=run_id,
                agent_name="checkpoint",
                role="handoff",
                content=json.dumps({
                    "kind": "manual_resume",
                    "next_agent": next_agent,
                    "paused_at": pause.paused_at,
                }),
            )
            run_control.clear_pause_state(conn, run_id)
            conn.execute(
                "UPDATE runs SET status = 'running', ended_at = NULL WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()

            final_status, note, handoffs = _drive_loop(
                conn=conn,
                cfg=cfg,
                log=log,
                run_id=run_id,
                project_id=project_id,
                starting_agent=next_agent,
                starting_payload=next_payload,
                max_handoffs=max_handoffs,
                max_tool_rounds=max_tool_rounds,
                enable_checkpoints=True,
                handoffs_so_far=handoffs_so_far,
                code_review_cycles_so_far=code_review_cycles_so_far,
            )
            end_run(conn, run_id, final_status, note)
            conn.commit()
            log.info("run.end", status=final_status, handoffs=handoffs, note=note)
            return run_id

        # --- Path 2: checkpoint-based resume -------------------------------
        cp = checkpoints.latest_checkpoint(conn, run_id)
        if cp is None:
            raise RuntimeError(f"run {run_id} is paused but has no checkpoint")

        # --- Path 2a: operator "re-run with nudge" -------------------------
        # The checkpoint was resolved with decision='rerun': re-invoke the
        # agent that produced it (cp.from_agent) on its original inbound plus
        # the operator's nudge, then let it pause again at the same stage.
        op = cp.operator_input or {}
        if cp.status != "pending" and op.get("decision") == checkpoints.DECISION_RERUN:
            from_agent = cp.from_agent
            nudge = op.get("instructions") or ""
            inbound = _latest_inbound_for_agent(conn, run_id, from_agent) or cp.default_payload
            starting_payload = _apply_nudge(inbound, nudge)
            log.info(
                "run.resume_rerun",
                checkpoint_id=cp.checkpoint_id,
                stage=cp.stage,
                rerun_agent=from_agent,
                nudge_chars=len(nudge),
            )
            record_message(
                conn,
                run_id=run_id,
                agent_name=from_agent or "checkpoint",
                role="handoff",
                content=json.dumps({
                    "checkpoint_id": cp.checkpoint_id,
                    "stage": cp.stage,
                    "decision": "rerun",
                    "next_agent": from_agent,
                    "nudge": nudge,
                }),
            )
            run_control.cancel_pause_request(conn, run_id)
            conn.execute(
                "UPDATE runs SET status = 'running', ended_at = NULL WHERE run_id = ?",
                (run_id,),
            )
            conn.commit()
            final_status, note, handoffs = _drive_loop(
                conn=conn,
                cfg=cfg,
                log=log,
                run_id=run_id,
                project_id=project_id,
                starting_agent=from_agent,
                starting_payload=starting_payload,
                max_handoffs=max_handoffs,
                max_tool_rounds=max_tool_rounds,
                enable_checkpoints=True,
                model_overrides=_model_overrides_from_op(op),
            )
            end_run(conn, run_id, final_status, note)
            conn.commit()
            log.info("run.end", status=final_status, handoffs=handoffs, note=note)
            return run_id

        if cp.status == "pending":
            raise RuntimeError(
                f"checkpoint {cp.checkpoint_id} is still pending — resolve it first"
            )
        if cp.status == "rejected":
            # Mark the run as cancelled and return its id; do not resume.
            end_run(conn, run_id, "cancelled", f"checkpoint {cp.checkpoint_id} rejected")
            conn.commit()
            return run_id

        next_payload = checkpoints.resolve_payload_for_resume(cp)
        next_agent = cp.to_agent

        log.info(
            "run.resume",
            checkpoint_id=cp.checkpoint_id,
            stage=cp.stage,
            decision=(cp.operator_input or {}).get("decision"),
            next_agent=next_agent,
        )

        # Audit trail: a synthetic handoff message documents the resume.
        record_message(
            conn,
            run_id=run_id,
            agent_name=cp.from_agent or "checkpoint",
            role="handoff",
            content=json.dumps(
                {
                    "checkpoint_id": cp.checkpoint_id,
                    "stage": cp.stage,
                    "decision": (cp.operator_input or {}).get("decision"),
                    "next_agent": next_agent,
                    "instructions": (cp.operator_input or {}).get("instructions"),
                }
            ),
        )

        # Pre-emptively clear any stale pause_requested flag — the operator
        # is explicitly continuing through a checkpoint, which overrides a
        # never-honoured prior Pause click.
        run_control.cancel_pause_request(conn, run_id)

        # If the operator approved a terminal checkpoint (next_agent == DONE),
        # there's nothing to drive — close out the run as completed.
        if next_agent == DONE or not next_agent:
            end_run(conn, run_id, "completed", f"resumed past terminal checkpoint {cp.checkpoint_id}")
            conn.commit()
            log.info("run.end", status="completed", handoffs=0, note="terminal checkpoint approved")
            return run_id

        # Flip status running while the loop drives.
        conn.execute(
            "UPDATE runs SET status = 'running', ended_at = NULL WHERE run_id = ?",
            (run_id,),
        )
        conn.commit()

        final_status, note, handoffs = _drive_loop(
            conn=conn,
            cfg=cfg,
            log=log,
            run_id=run_id,
            project_id=project_id,
            starting_agent=next_agent,
            starting_payload=next_payload,
            max_handoffs=max_handoffs,
            max_tool_rounds=max_tool_rounds,
            enable_checkpoints=True,
            model_overrides=_model_overrides_from_op(cp.operator_input),
        )

        end_run(conn, run_id, final_status, note)
        conn.commit()
        log.info("run.end", status=final_status, handoffs=handoffs, note=note)
        return run_id
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="autoscientist")
    parser.add_argument(
        "--agent",
        help="Starting agent name (requires prompts/<name>.md and an entry in models.toml)",
    )
    parser.add_argument(
        "--project",
        default="adhoc",
        help="Project ID for grouping runs (default: 'adhoc')",
    )
    parser.add_argument(
        "--payload",
        default="",
        help="Initial payload to feed the starting agent",
    )
    parser.add_argument(
        "--max-handoffs",
        type=int,
        default=None,
        help="Override default max_handoffs from config/default.toml",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        help="Resume a paused run from its most recently resolved checkpoint.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable checkpoint pauses (substrate testing only — production runs must keep them on).",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    max_handoffs = args.max_handoffs or cfg.default.get("runtime", {}).get("default_max_handoffs", 50)

    if args.resume:
        run_id = resume_run(args.resume, max_handoffs=max_handoffs, cfg=cfg)
    else:
        if not args.agent:
            parser.error("either --agent or --resume RUN_ID is required")
        run_id = run(
            starting_agent=args.agent,
            project_id=args.project,
            initial_payload=args.payload,
            max_handoffs=max_handoffs,
            enable_checkpoints=not args.no_checkpoints,
            cfg=cfg,
        )
    print(run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
