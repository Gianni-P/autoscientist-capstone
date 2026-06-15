"""Smoke test: prove a tool call round-trips through the Ollama OpenAI-compat shim.

Item 1 of the 2026-05-31 audit. The cost runaway traced to code_gen/test_gen
spending on Claude because the local Qwen tool-calling leg was believed broken.
This script proves tool-calling works *locally, through clients/ollama.py* (the
production shim, not a raw curl) against the running Ollama server, for one or
more models. It does a full two-round trip:

    round 1: ask the model to use a tool      -> expect a well-formed tool_call
    round 2: feed the tool result back        -> expect a final answer

Run inside WSL (the venv is Linux; Ollama binds on localhost:11434 there):

    ./.venv/bin/python scripts/smoke_local_toolcall.py
    ./.venv/bin/python scripts/smoke_local_toolcall.py qwen2.5:32b   # subset

Exits non-zero if any model fails either round. This is intentionally
network-free of Anthropic — it never touches the budget ledger or the API key,
so it is safe to run repeatedly at zero cost.
"""

from __future__ import annotations

import json
import sys

from autoscientist.clients import ollama

# OpenAI tool-calling shape, same shape tools/registry.openai_schemas emits.
MULTIPLY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers and return the product.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }
]


def round_trip(model: str, *, no_think: bool) -> tuple[bool, str]:
    """Two-round tool-use exchange. Returns (passed, human-readable detail)."""
    user = (
        "What is 12 multiplied by 7? You MUST call the multiply tool to compute "
        "it, then state the product."
    )
    messages: list[dict] = [{"role": "user", "content": user}]

    r1 = ollama.complete(
        model=model, system=None, messages=messages,
        max_tokens=2048, temperature=0.0, tools=MULTIPLY_TOOL, no_think=no_think,
    )
    if not r1.tool_calls:
        return False, f"round1: no tool_calls (content={r1.content[:160]!r})"
    tc = r1.tool_calls[0]
    if tc.name != "multiply" or tc.input.get("a") != 12 or tc.input.get("b") != 7:
        return False, f"round1: wrong/garbled tool call: {tc.name} args={tc.input}"

    # Feed the tool result back in OpenAI shape, exactly as runner._invoke_agent does.
    messages.append(r1.raw_content_blocks)
    messages.append(
        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"product": 84})}
    )
    r2 = ollama.complete(
        model=model, system=None, messages=messages,
        max_tokens=2048, temperature=0.0, tools=MULTIPLY_TOOL, no_think=no_think,
    )
    if "84" not in (r2.content or ""):
        return False, f"round2: final answer missing '84' (content={r2.content[:200]!r})"
    return True, (
        f"tool_call(a={tc.input.get('a')},b={tc.input.get('b')}) "
        f"-> final={r2.content.strip()[:80]!r}"
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    models = argv or ["qwen3.6:27b", "qwen2.5:32b"]
    all_ok = True
    for m in models:
        try:
            passed, detail = round_trip(m, no_think=True)
        except Exception as e:  # noqa: BLE001 - smoke test surfaces any failure
            passed, detail = False, f"EXCEPTION {type(e).__name__}: {e}"
        print(f"[{'PASS' if passed else 'FAIL'}] {m}: {detail}")
        all_ok = all_ok and passed
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
