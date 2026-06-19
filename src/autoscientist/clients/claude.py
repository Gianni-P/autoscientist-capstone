"""Anthropic Claude client adapter.

Wraps the ``anthropic`` SDK with:
  * retry on transient errors (RateLimit + 5xx + connection) via tenacity
  * usage-token extraction into a provider-agnostic :class:`CompletionResult`
  * structured logging (API key never logged)

Use via :func:`complete`. The router is the only intended call site
outside tests.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import structlog
from anthropic import (
    Anthropic,
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from autoscientist.clients.base import CompletionResult, ToolCall

log = structlog.get_logger("autoscientist.clients.claude")
_tenacity_log = logging.getLogger("autoscientist.clients.claude.tenacity")

_RETRYABLE = (RateLimitError, APIConnectionError, InternalServerError)

_client: Anthropic | None = None


class MissingApiKey(RuntimeError):
    """Raised when ANTHROPIC_API_KEY is not configured."""


def _get_client() -> Anthropic:
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingApiKey(
            "ANTHROPIC_API_KEY is not set. "
            "Add `export ANTHROPIC_API_KEY=sk-ant-...` to ~/.bashrc and source it, "
            "or copy .env.example to .env and fill it in."
        )
    _client = Anthropic(api_key=api_key)
    return _client


def reset_client_for_tests() -> None:
    """Drop the cached client. Used by tests that mutate ANTHROPIC_API_KEY."""
    global _client
    _client = None


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, max=60),
    reraise=True,
    before_sleep=before_sleep_log(_tenacity_log, logging.WARNING),
)
def complete(
    *,
    model: str,
    system: str | None,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> CompletionResult:
    """Single chat completion against Claude with retries on transient errors.

    ``tools`` is a list of Anthropic-shape tool schemas (see
    ``tools/registry.py:anthropic_schemas``). When provided, the model may
    emit ``tool_use`` content blocks; these are surfaced as
    :class:`ToolCall` entries on :class:`CompletionResult`.
    """
    client = _get_client()
    log.info(
        "claude.complete.start",
        model=model,
        n_messages=len(messages),
        max_tokens=max_tokens,
        temperature=temperature,
        n_tools=len(tools) if tools else 0,
    )

    kwargs: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools
        # Prompt caching (top-level auto-caching, anthropic>=0.100): place an
        # ephemeral cache breakpoint on the last cacheable block each call. In an
        # agentic tool loop the stable prefix (system + the large first user
        # payload + prior rounds) is re-sent every round, so round k+1 reads it
        # at ~0.1x instead of full input price — this is the bulk of code_review
        # spend (~1.4M input tok/run, mostly the re-sent source). Gated on
        # `tools` because only looping agents re-send a prefix; single-shot
        # agents would pay the 1.25x write with no read. Caching never changes
        # outputs, only cost. Cache usage is folded into cost_usd in router.py.
        kwargs["cache_control"] = {"type": "ephemeral"}
    if extra_headers:
        kwargs["extra_headers"] = extra_headers

    try:
        response = client.messages.create(**kwargs)
    except BadRequestError as e:
        # Self-heal model-capability drift: newer Claude models may reject
        # parameters older ones accepted (observed 2026-05-31: opus-4-7 returns
        # "temperature is deprecated for this model"). Drop the offending
        # parameter and retry once rather than failing the whole run. The
        # router also drops temperature proactively via supports_temperature;
        # this catches models not yet flagged.
        msg = str(e).lower()
        _param_rejected = any(
            w in msg for w in
            ("deprecat", "unsupported", "not support", "not allowed", "invalid", "does not support")
        )
        if "temperature" in kwargs and "temperature" in msg and _param_rejected:
            log.warning("claude.temperature_unsupported_retry", model=model)
            kwargs.pop("temperature", None)
            response = client.messages.create(**kwargs)
        else:
            raise

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    raw_blocks: list[dict[str, Any]] = []
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            text_parts.append(text)
            raw_blocks.append({"type": "text", "text": text})
        elif btype == "tool_use":
            tu_id = getattr(block, "id", "")
            tu_name = getattr(block, "name", "")
            tu_input = getattr(block, "input", {}) or {}
            tool_calls.append(ToolCall(id=tu_id, name=tu_name, input=dict(tu_input)))
            raw_blocks.append({
                "type": "tool_use",
                "id": tu_id,
                "name": tu_name,
                "input": dict(tu_input),
            })
        else:
            # Unknown block type — preserve text if any, log for visibility.
            block_text = getattr(block, "text", None)
            if block_text:
                text_parts.append(block_text)
            log.warning("claude.unknown_block_type", btype=btype)
    content = "\n".join(text_parts)

    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    result = CompletionResult(
        content=content,
        model=response.model,
        provider="claude",
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        finish_reason=response.stop_reason,
        tool_calls=tool_calls,
        raw_content_blocks=raw_blocks,
        raw=response,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
    log.info(
        "claude.complete.done",
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        finish_reason=result.finish_reason,
        n_tool_calls=len(tool_calls),
    )
    return result
