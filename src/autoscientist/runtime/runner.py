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
import sys
import time
from pathlib import Path

import structlog

from autoscientist.agents import get_agent as registry_get_agent
from autoscientist.checkpoints import manager as checkpoints
from autoscientist.clients.router import route
from autoscientist.runtime import control as run_control
from autoscientist.runtime.agent import Agent, load_prompt
from autoscientist.runtime.config import Config, load_config
from autoscientist.runtime.handoff import DONE, Handoff, parse_handoff
from autoscientist.runtime.payload_files import (
    build_code_review_payload_from_sandbox,
    persist_files_from_payload,
)
from autoscientist.state.db import (
    end_run,
    open_db,
    record_message,
    start_run,
)
from autoscientist.tools import registry as tool_registry

DEFAULT_MAX_TOOL_ROUNDS = 40
DEFAULT_MAX_CODE_REVIEW_CYCLES = 3

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
    "code_gen": "test_gen",
    "test_gen": "code_review",
    "code_review": "results_validator",
    "results_validator": "paper_writer",
    "paper_writer": "peer_reviewer",
}

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
    """Return per-agent max_tool_rounds from models.toml, or ``default``."""
    agents = cfg.models.get("agents", {})
    agent_cfg = agents.get(agent_name, {})
    return int(agent_cfg.get("max_tool_rounds", default))


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
    if "invocation_ceiling_usd" in agent_cfg:
        return float(agent_cfg["invocation_ceiling_usd"])
    budget = cfg.models.get("budget", {})
    if "default_invocation_ceiling_usd" in budget:
        return float(budget["default_invocation_ceiling_usd"])
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
):
    """Run one agent invocation including its tool-use loop.

    Returns the final assistant ``CompletionResult`` (the one with no tool
    calls; its ``content`` carries any ``HANDOFF: <target>`` directive).

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

    log.warning(
        "run.tool_loop_max_rounds_reached", agent=agent.name, rounds=max_tool_rounds,
    )
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
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = text[start:i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
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
    handoffs_so_far: int = 0,
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

    # Cap on consecutive code_review firings in this _drive_loop pass.
    # The counter is local to this invocation; resume_run starts a fresh
    # _drive_loop with code_review_cycles=0, so a pause naturally resets
    # the cap after the operator approves at CP3.
    code_review_cycles = 0
    max_code_review_cycles = _max_code_review_cycles(cfg)

    try:
        while True:
            agent = _build_agent(cfg, current_agent_name)
            prompt = load_prompt(agent.system_prompt_path)
            inbound_text = current_payload or "(no payload)"

            # code_review has no file-reading tool, so an empty handoff payload
            # (upstream code_gen/test_gen exhausted its tool rounds and emitted
            # no content — a recurring qwen3-coder failure mode) would make the
            # review a no-op and open a degenerate CP3 carrying a "(no payload)"
            # complaint. Rebuild its input from the sandbox on disk so the
            # review always runs against the real code/tests. Single guard here
            # covers every path that can feed code_review an empty payload
            # (forced handoff, resume, manual re-entry).
            if current_agent_name == "code_review" and (
                not current_payload or not current_payload.strip()
            ):
                projects_root = cfg.root / cfg.default.get("paths", {}).get(
                    "projects_dir", "projects"
                )
                rebuilt = build_code_review_payload_from_sandbox(
                    project_id=project_id, projects_root=projects_root,
                )
                if rebuilt:
                    inbound_text = rebuilt
                    log.warning(
                        "run.code_review_payload_reconstructed",
                        agent=current_agent_name,
                        chars=len(rebuilt),
                    )

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
            )

            if current_agent_name == "code_review":
                code_review_cycles += 1

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
                log.warning(
                    "run.handoff_off_topology",
                    from_agent=current_agent_name,
                    to_agent=handoff.to_agent,
                    allowed=list(agent.handoff_targets),
                )

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
        )

        end_run(conn, run_id, final_status, note)
        conn.commit()
        log.info("run.end", status=final_status, handoffs=handoffs, note=note)
        return run_id
    finally:
        conn.close()


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
            )
            end_run(conn, run_id, final_status, note)
            conn.commit()
            log.info("run.end", status=final_status, handoffs=handoffs, note=note)
            return run_id

        # --- Path 2: checkpoint-based resume -------------------------------
        cp = checkpoints.latest_checkpoint(conn, run_id)
        if cp is None:
            raise RuntimeError(f"run {run_id} is paused but has no checkpoint")
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
