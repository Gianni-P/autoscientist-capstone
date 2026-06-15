"""Shared types and cost helpers for LLM client adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single LLM-issued tool invocation request.

    Provider-agnostic. Both Anthropic ``tool_use`` blocks and OpenAI
    ``tool_calls`` entries are normalized into this shape so the runner's
    tool-loop is provider-independent.
    """

    id: str  # provider-issued id, used to round-trip the result
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResult:
    """Provider-agnostic result of a single chat completion call."""

    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None = None
    reasoning: str | None = None  # set by Ollama for thinking models
    # Real USD charged for THIS call, set by clients/router.route after the
    # cache/dispatch decision (0.0 for a cache hit, actual usage-based cost
    # otherwise). None means "not priced by the router" (e.g. a provider
    # adapter called directly). The runner sums this across a single agent
    # invocation to enforce the per-invocation budget cap.
    cost_usd: float | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Original assistant content blocks (Anthropic) or message dict (OpenAI),
    # preserved so the runner can append them to ``messages`` for round-tripping.
    raw_content_blocks: Any = field(default=None, repr=False)
    raw: Any = field(default=None, repr=False)
    # Anthropic prompt-cache usage (0 for providers/calls without caching).
    # cache_read bills at ~0.1x the prompt rate, cache_write at ~1.25x; the
    # router folds these into cost_usd so the budget guardrail reflects real
    # spend even though usage.input_tokens excludes the cached prefix.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


def call_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_usd_per_mtok: float,
    output_usd_per_mtok: float,
) -> float:
    """Cost of one completion call given token counts.

    For a *pre-check* (worst-case) estimate, pass the prompt-token estimate
    and ``max_tokens`` (the worst-case completion size). For the *actual*
    cost, pass the ``usage`` token counts returned by the API.
    """
    return (
        (prompt_tokens / 1_000_000.0) * prompt_usd_per_mtok
        + (completion_tokens / 1_000_000.0) * output_usd_per_mtok
    )


def estimate_prompt_tokens(system: str | None, messages: list[dict[str, Any]]) -> int:
    """Rough char/4 estimate of prompt tokens for budget gating.

    Anthropic's ``messages.count_tokens`` is more accurate but is an API
    call. For pre-check budgeting a 30%-loose heuristic is fine — it
    biases conservative (chars-per-token is closer to 3.5 for code).
    """
    total_chars = len(system or "")
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                text = block.get("text") if isinstance(block, dict) else None
                if text:
                    total_chars += len(text)
    return max(1, total_chars // 4)
