"""Ollama OpenAI-compatible client adapter.

Targets local Ollama at ``/v1``. Configured per-model via ``models.toml``.

qwen3.6:27b emits chain-of-thought on a separate ``reasoning`` field
(an Ollama extension to the OpenAI shape) which we capture and persist
in JSONL logs. When the model alias has ``no_think = true``, the system
prompt is suffixed with ``/no_think`` to suppress reasoning entirely.

Reasoning loop detection: if the model produces 0 content chars but
accumulates long reasoning with repeated blocks (the "planning
oscillation" pattern), a single retry with ``no_think=True`` is
attempted. This breaks the degenerate loop observed with Qwen on
complex code-gen tasks.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import openai
import structlog
from openai import OpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from autoscientist.clients.base import CompletionResult, ToolCall

log = structlog.get_logger("autoscientist.clients.ollama")
_tenacity_log = logging.getLogger("autoscientist.clients.ollama.tenacity")

_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_client_cache: dict[str, OpenAI] = {}


def _get_client(base_url: str | None = None) -> OpenAI:
    base = base_url or os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
    if base not in _client_cache:
        # Ollama ignores the API key but the OpenAI SDK requires non-empty.
        _client_cache[base] = OpenAI(base_url=base, api_key="ollama", timeout=180.0)
    return _client_cache[base]


def _detect_reasoning_loop(reasoning: str, window: int = 400) -> bool:
    """Return True if ``reasoning`` contains repeated blocks.

    The detection heuristic: take the last ``window`` chars of reasoning
    and check if that exact block appears at least twice earlier. This
    catches the Qwen "Actually, let me think..." planning oscillation
    without false-positives on legitimate multi-paragraph reasoning.
    """
    if len(reasoning) < window * 3:
        return False
    tail = reasoning[-window:]
    # Count non-overlapping occurrences of the tail in the full text
    count = reasoning.count(tail)
    return count >= 3


def reset_client_for_tests() -> None:
    _client_cache.clear()


@retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=30),
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
    no_think: bool = False,
    base_url: str | None = None,
) -> CompletionResult:
    """Single chat completion against Ollama via OpenAI-compatible /v1.

    ``tools`` is OpenAI tool-calling shape (see
    ``tools/registry.py:openai_schemas``). When provided, ``choice.message
    .tool_calls`` may carry one or more entries; these are normalized to
    :class:`ToolCall`. The tool-result round-trip uses ``role: "tool"``
    messages keyed by ``tool_call_id`` (callers do this in the runner).
    """
    client = _get_client(base_url)

    chat_messages: list[dict[str, Any]] = []
    sys_text = (system or "").rstrip()
    if no_think:
        sys_text = f"{sys_text}\n\n/no_think".lstrip() if sys_text else "/no_think"
    if sys_text:
        chat_messages.append({"role": "system", "content": sys_text})
    chat_messages.extend(messages)

    log.info(
        "ollama.complete.start",
        model=model,
        n_messages=len(chat_messages),
        max_tokens=max_tokens,
        no_think=no_think,
        temperature=temperature,
        n_tools=len(tools) if tools else 0,
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    message = choice.message
    content = message.content or ""

    # Ollama emits chain-of-thought as a `reasoning` field on the message.
    # The OpenAI SDK passes unknown fields through model_extra in pydantic v2.
    reasoning: str | None = None
    extra = getattr(message, "model_extra", None) or {}
    if "reasoning" in extra and extra["reasoning"]:
        reasoning = extra["reasoning"]
    elif getattr(message, "reasoning", None):
        reasoning = message.reasoning  # type: ignore[attr-defined]

    # Normalize OpenAI tool_calls -> ToolCall list. arguments is a JSON string.
    import json as _json
    tool_calls: list[ToolCall] = []
    raw_msg_dict: dict[str, Any] = {"role": "assistant", "content": content}
    raw_tool_calls: list[dict[str, Any]] = []
    for tc in (getattr(message, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", "") if fn is not None else ""
        args_raw = getattr(fn, "arguments", "") if fn is not None else ""
        try:
            args = _json.loads(args_raw) if args_raw else {}
        except _json.JSONDecodeError:
            args = {"_raw_arguments": args_raw}
        tc_id = getattr(tc, "id", "")
        tool_calls.append(ToolCall(id=tc_id, name=name, input=args))
        raw_tool_calls.append({
            "id": tc_id,
            "type": "function",
            "function": {"name": name, "arguments": args_raw or _json.dumps(args)},
        })
    if raw_tool_calls:
        raw_msg_dict["tool_calls"] = raw_tool_calls

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    result = CompletionResult(
        content=content,
        model=response.model,
        provider="ollama",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=choice.finish_reason,
        reasoning=reasoning,
        tool_calls=tool_calls,
        raw_content_blocks=raw_msg_dict,
        raw=response,
    )
    log.info(
        "ollama.complete.done",
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        reasoning_chars=len(reasoning) if reasoning else 0,
        finish_reason=result.finish_reason,
        n_tool_calls=len(tool_calls),
    )

    # Reasoning loop detection: if the model produced 0 content but long
    # reasoning with repeated blocks, retry once with thinking suppressed.
    if (
        not no_think
        and not content
        and not tool_calls
        and reasoning
        and _detect_reasoning_loop(reasoning)
    ):
        log.warning(
            "ollama.reasoning_loop_detected",
            model=model,
            reasoning_chars=len(reasoning),
        )
        return complete(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            no_think=True,  # force thinking off for retry
            base_url=base_url,
        )

    return result
