"""Model router with cache and budget gating.

The router is the only call site for ``clients/claude.py`` and
``clients/ollama.py`` in production code. It encapsulates:

  1. cache check (hit -> free; charge $0 ledger entry)
  2. cost estimate (upper bound using ``max_tokens``)
  3. per-call cost ceiling (``cost_ceiling_usd`` per agent)
  4. monthly budget refusal (``cap - buffer``)
  5. provider dispatch
  6. usage-based charge recording
  7. cache write

Inputs come from ``config/models.toml``.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

import structlog

from autoscientist.clients import claude as claude_client
from autoscientist.clients import ollama as ollama_client
from autoscientist.clients.base import (
    CompletionResult,
    call_cost_usd,
    estimate_prompt_tokens,
)
from autoscientist.clients.cache import cache_key, get_cached, put_cached
from autoscientist.runtime.budget import (
    BudgetConfig,
    assert_can_spend,
    assert_project_budget,
    record_charge,
)
from autoscientist.runtime.config import Config, load_config

log = structlog.get_logger("autoscientist.clients.router")


class CostCeilingExceeded(RuntimeError):
    """Per-call cost ceiling exceeded for an agent."""


class UnknownAgent(KeyError):
    pass


class UnknownModel(KeyError):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    alias: str
    provider: str
    model_id: str
    prompt_usd_per_mtok: float
    output_usd_per_mtok: float
    default_max_tokens: int
    no_think: bool
    supports_temperature: bool = True


@dataclass(frozen=True)
class ResolvedAgent:
    name: str
    model: ResolvedModel
    cost_ceiling_usd: float


def _resolve_model(cfg: Config, alias: str) -> ResolvedModel:
    models = cfg.models.get("models", {})
    if alias not in models:
        raise UnknownModel(f"unknown model alias: {alias}")
    m = models[alias]
    return ResolvedModel(
        alias=alias,
        provider=m["provider"],
        model_id=m["model_id"],
        prompt_usd_per_mtok=float(m.get("prompt_usd_per_mtok", 0.0)),
        output_usd_per_mtok=float(m.get("output_usd_per_mtok", 0.0)),
        default_max_tokens=int(m.get("default_max_tokens", 4096)),
        no_think=bool(m.get("no_think", False)),
        supports_temperature=bool(m.get("supports_temperature", True)),
    )


def resolve_agent(cfg: Config, agent_name: str) -> ResolvedAgent:
    agents = cfg.models.get("agents", {})
    if agent_name not in agents:
        raise UnknownAgent(f"unknown agent: {agent_name}")
    a = agents[agent_name]
    return ResolvedAgent(
        name=agent_name,
        model=_resolve_model(cfg, a["model"]),
        cost_ceiling_usd=float(a.get("cost_ceiling_usd", 0.0)),
    )


def _budget_config(cfg: Config) -> BudgetConfig:
    return BudgetConfig.from_dict(cfg.models.get("budget", {}))


def _ollama_base_url(cfg: Config) -> str | None:
    p = cfg.models.get("providers", {}).get("ollama", {})
    return os.environ.get(p.get("base_url_env", "OLLAMA_BASE_URL"), p.get("base_url_default"))


def route(
    *,
    conn: sqlite3.Connection,
    agent_name: str,
    system: str | None,
    messages: list[dict[str, Any]],
    run_id: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    tools_anthropic: list[dict[str, Any]] | None = None,
    tools_openai: list[dict[str, Any]] | None = None,
    tools_signature: str | None = None,
    cfg: Config | None = None,
    project_id: str | None = None,
) -> CompletionResult:
    """Dispatch a single LLM call. Provider-aware tool plumbing.

    ``tools_anthropic`` and ``tools_openai`` are pre-rendered schema lists
    (callers use ``tools/registry.py`` to build them). The router picks the
    one matching the resolved provider; both are accepted so callers don't
    have to know the provider.
    """
    cfg = cfg or load_config()
    agent = resolve_agent(cfg, agent_name)
    model = agent.model
    if max_tokens is None:
        max_tokens = model.default_max_tokens
    # Some models (e.g. claude-opus-4-7) reject the `temperature` parameter.
    # Drop it here so a prompt's frontmatter temperature doesn't 400 the call.
    if not model.supports_temperature and temperature is not None:
        log.info("router.temperature_dropped", agent=agent_name, model=model.model_id)
        temperature = None
    bcfg = _budget_config(cfg)

    key = cache_key(
        provider=model.provider,
        model=model.model_id,
        system=system,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools_signature=tools_signature,
        extra={"no_think": model.no_think} if model.provider == "ollama" else None,
    )

    cached = get_cached(conn, key)
    if cached is not None:
        blob = cached.response_blob
        # tool_calls were stored as plain dicts; rehydrate.
        from autoscientist.clients.base import ToolCall

        tcs_raw = blob.get("tool_calls") or []
        tcs = [
            ToolCall(id=t["id"], name=t["name"], input=dict(t.get("input") or {}))
            for t in tcs_raw
        ]
        result = CompletionResult(
            content=blob.get("content", ""),
            model=blob.get("model", model.model_id),
            provider=model.provider,
            prompt_tokens=cached.prompt_tokens or 0,
            completion_tokens=cached.completion_tokens or 0,
            finish_reason=blob.get("finish_reason"),
            reasoning=blob.get("reasoning"),
            tool_calls=tcs,
            raw_content_blocks=blob.get("raw_content_blocks"),
            raw=None,
            cost_usd=0.0,  # cache hits are free; matches the $0 ledger row below
        )
        record_charge(
            conn,
            run_id=run_id, agent_name=agent_name,
            provider=model.provider, model=model.model_id,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=0.0, cache_hit=True,
        )
        log.info(
            "router.cache_hit",
            agent=agent_name, model=model.model_id, key=key[:12],
        )
        return result

    prompt_estimate = estimate_prompt_tokens(system, messages)
    estimated_cost = call_cost_usd(
        prompt_tokens=prompt_estimate,
        completion_tokens=max_tokens,
        prompt_usd_per_mtok=model.prompt_usd_per_mtok,
        output_usd_per_mtok=model.output_usd_per_mtok,
    )

    if agent.cost_ceiling_usd > 0 and estimated_cost > agent.cost_ceiling_usd:
        raise CostCeilingExceeded(
            f"agent {agent_name}: estimated ${estimated_cost:.4f} exceeds ceiling "
            f"${agent.cost_ceiling_usd:.4f}"
        )

    if estimated_cost > 0:
        assert_can_spend(conn, bcfg, estimated_cost)

    # Per-project soft cap (reads project config.toml if project_id given).
    if project_id and estimated_cost > 0:
        project_cfg_path = cfg.root / "projects" / project_id / "config.toml"
        if project_cfg_path.exists():
            import tomllib
            with project_cfg_path.open("rb") as f:
                pcfg = tomllib.load(f)
            soft_cap = float(pcfg.get("budget", {}).get("project_soft_cap_usd", 0))
            if soft_cap > 0:
                assert_project_budget(conn, project_id, soft_cap, estimated_cost)

    started = time.monotonic()
    if model.provider == "claude":
        result = claude_client.complete(
            model=model.model_id, system=system, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            tools=tools_anthropic,
        )
    elif model.provider == "ollama":
        result = ollama_client.complete(
            model=model.model_id, system=system, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            tools=tools_openai,
            no_think=model.no_think,
            base_url=_ollama_base_url(cfg),
        )
    elif model.provider == "mock":
        from autoscientist.clients import mock as mock_client
        # Mock accepts either schema; pass whichever was built.
        mock_tools = tools_anthropic or tools_openai
        result = mock_client.complete(
            agent_name=agent_name, model=model.model_id, system=system,
            messages=messages, max_tokens=max_tokens, temperature=temperature,
            tools=mock_tools,
        )
    else:
        raise UnknownModel(f"unsupported provider: {model.provider}")
    elapsed_ms = int((time.monotonic() - started) * 1000)

    actual_cost = call_cost_usd(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        prompt_usd_per_mtok=model.prompt_usd_per_mtok,
        output_usd_per_mtok=model.output_usd_per_mtok,
    )
    # Prompt-cache billing: result.prompt_tokens (usage.input_tokens) is the
    # UNCACHED remainder. Anthropic bills cache reads at ~0.1x and cache writes
    # at ~1.25x the prompt rate; add them so the per-invocation cap and monthly
    # ledger reflect real spend rather than under-counting the cached prefix.
    if result.cache_read_tokens or result.cache_write_tokens:
        actual_cost += (
            (result.cache_read_tokens * 0.1 + result.cache_write_tokens * 1.25)
            / 1_000_000.0
        ) * model.prompt_usd_per_mtok
    # Stamp the real cost on the result so the runner can sum it across a
    # single agent invocation (per-invocation budget cap in _invoke_agent).
    result.cost_usd = actual_cost

    record_charge(
        conn,
        run_id=run_id, agent_name=agent_name,
        provider=model.provider, model=model.model_id,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=actual_cost, cache_hit=False,
    )

    request_blob = {
        "system": system, "messages": messages, "model": model.model_id,
        "max_tokens": max_tokens, "temperature": temperature,
        "tools_signature": tools_signature,
    }
    response_blob = {
        "content": result.content,
        "model": result.model,
        "finish_reason": result.finish_reason,
        "reasoning": result.reasoning,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "input": tc.input} for tc in result.tool_calls
        ],
        "raw_content_blocks": result.raw_content_blocks,
    }
    put_cached(
        conn, key=key, provider=model.provider, model=model.model_id,
        request_blob=request_blob, response_blob=response_blob,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )

    log.info(
        "router.dispatched",
        agent=agent_name, model=model.model_id, provider=model.provider,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=round(actual_cost, 6),
        latency_ms=elapsed_ms,
    )
    return result
