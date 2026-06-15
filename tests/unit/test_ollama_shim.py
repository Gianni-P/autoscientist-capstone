"""Unit tests for the Ollama OpenAI-compat shim's tool-call normalization.

Network-free: we monkeypatch ``ollama._get_client`` to return a fake client so
these assert the shim's parsing logic deterministically, without a live server.

The adversarial case (2026-05-31 audit, item 1): a model can return tool-call
``arguments`` that are not valid JSON. The shim must not crash — it must surface
the raw string under ``_raw_arguments`` so the runner can still feed a
tool_result back and keep the conversation moving.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from autoscientist.clients import ollama


def _fake_tool_call(name, arguments, *, call_id="call_x"):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(id=call_id, function=fn)


def _fake_response(*, content="", tool_calls=None, finish_reason="tool_calls",
                   model="qwen2.5:32b"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [], reasoning=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


def _install_fake_client(monkeypatch, response):
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: response)
        )
    )
    monkeypatch.setattr(ollama, "_get_client", lambda base_url=None: client)


def _complete():
    return ollama.complete(
        model="qwen2.5:32b",
        system=None,
        messages=[{"role": "user", "content": "x"}],
        max_tokens=128,
        temperature=0.0,
        tools=[{"type": "function", "function": {"name": "multiply", "parameters": {}}}],
        no_think=True,  # avoid the reasoning-loop retry branch
    )


def test_malformed_tool_arguments_do_not_crash(monkeypatch):
    """Non-JSON arguments → surfaced under _raw_arguments, no exception."""
    _install_fake_client(
        monkeypatch,
        _fake_response(tool_calls=[_fake_tool_call("write_file", "{not valid json")]),
    )
    result = _complete()
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "write_file"
    assert tc.input == {"_raw_arguments": "{not valid json"}
    # The raw assistant message must still round-trip the original arg string.
    assert result.raw_content_blocks["tool_calls"][0]["function"]["arguments"] == "{not valid json"


def test_valid_tool_arguments_parse(monkeypatch):
    _install_fake_client(
        monkeypatch,
        _fake_response(tool_calls=[_fake_tool_call("multiply", '{"a": 12, "b": 7}')]),
    )
    result = _complete()
    assert result.tool_calls[0].input == {"a": 12, "b": 7}
    assert result.finish_reason == "tool_calls"


def test_empty_arguments_become_empty_dict(monkeypatch):
    _install_fake_client(
        monkeypatch,
        _fake_response(tool_calls=[_fake_tool_call("list_sandbox", "")]),
    )
    result = _complete()
    assert result.tool_calls[0].input == {}


def test_missing_function_object_does_not_crash(monkeypatch):
    """A tool_call whose .function is None must yield name='' (caller rejects it)."""
    bad = SimpleNamespace(id="c1", function=None)
    _install_fake_client(monkeypatch, _fake_response(tool_calls=[bad]))
    result = _complete()
    assert result.tool_calls[0].name == ""
    assert result.tool_calls[0].input == {}


def test_content_only_no_tool_calls(monkeypatch):
    _install_fake_client(
        monkeypatch,
        _fake_response(content="done", tool_calls=[], finish_reason="stop"),
    )
    result = _complete()
    assert result.content == "done"
    assert result.tool_calls == []


if __name__ == "__main__":  # allow ./.venv/bin/python tests/unit/test_ollama_shim.py
    raise SystemExit(pytest.main([__file__, "-q"]))
