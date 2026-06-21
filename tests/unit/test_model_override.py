"""Per-leg model override: router resolution, checkpoint persistence, helpers.

Covers the operator-selected "choose the model for the next leg" feature:
  * router.route(model_override=...) dispatches to the chosen model and falls
    back safely on an unknown alias;
  * checkpoints.manager.resolve persists the picks in operator_input and the
    runner reads them back (dropping empty selections);
  * checkpoints.manager.next_leg_agents scopes the picker to the right agents;
  * web.queries.validate_model_overrides accepts/rejects correctly.

Network-free: the provider clients are monkeypatched; only the real router /
budget / DB code runs.
"""

from __future__ import annotations

import pytest

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.clients import router
from autoscientist.clients.base import CompletionResult
from autoscientist.runtime import runner
from autoscientist.runtime.config import load_config
from autoscientist.state.db import open_db, start_run
from autoscientist.web import queries


def _stub_providers(monkeypatch, captured):
    def fake_claude(**kw):
        captured["claude_model"] = kw["model"]
        return CompletionResult(
            content="HANDOFF: DONE", model=kw["model"], provider="claude",
            prompt_tokens=10, completion_tokens=5, finish_reason="end_turn",
        )

    def fake_ollama(**kw):
        captured["ollama_model"] = kw["model"]
        return CompletionResult(
            content="HANDOFF: DONE", model=kw["model"], provider="ollama",
            prompt_tokens=10, completion_tokens=5, finish_reason="end_turn",
        )

    monkeypatch.setattr(router.claude_client, "complete", fake_claude)
    monkeypatch.setattr(router.ollama_client, "complete", fake_ollama)


def _conn_run(tmp_path):
    conn = open_db(str(tmp_path / "mo.db"))
    run_id = start_run(conn, project_id="moproj", config_snapshot={})
    conn.commit()
    return conn, run_id


def test_route_uses_model_override(tmp_path, monkeypatch):
    """code_worker is local (qwen25_32b); overriding to claude_haiku must
    dispatch to the Claude client with Haiku's model id, not Ollama."""
    captured: dict[str, str] = {}
    _stub_providers(monkeypatch, captured)
    cfg = load_config(reload=True)
    conn, run_id = _conn_run(tmp_path)
    res = router.route(
        conn=conn, agent_name="code_worker", system="s",
        messages=[{"role": "user", "content": "hi"}],
        run_id=run_id, model_override="claude_haiku", cfg=cfg,
    )
    conn.close()
    assert captured.get("claude_model") == "claude-haiku-4-5-20251001"
    assert "ollama_model" not in captured  # configured (local) model was NOT used
    assert res.model == "claude-haiku-4-5-20251001"


def test_route_unknown_override_falls_back(tmp_path, monkeypatch):
    """An unknown alias is ignored — the agent's configured model is used."""
    captured: dict[str, str] = {}
    _stub_providers(monkeypatch, captured)
    cfg = load_config(reload=True)
    conn, run_id = _conn_run(tmp_path)
    res = router.route(
        conn=conn, agent_name="code_worker", system="s",
        messages=[{"role": "user", "content": "hi"}],
        run_id=run_id, model_override="does_not_exist", cfg=cfg,
    )
    conn.close()
    assert captured.get("ollama_model") == "qwen2.5-32b-64k"  # config default
    assert "claude_model" not in captured
    assert res.model == "qwen2.5-32b-64k"


def test_resolve_persists_model_overrides(tmp_path):
    """resolve() records the operator's picks in operator_input.model_overrides."""
    conn, run_id = _conn_run(tmp_path)
    cp_id = cp_manager.open_checkpoint(
        conn, run_id=run_id, stage=2, from_agent="methodology",
        to_agent="code_gen", agent_output_raw="{}", default_payload="{}",
    )
    conn.commit()
    rec = cp_manager.resolve(
        conn, checkpoint_id=cp_id, decision="approve",
        model_overrides={"code_gen": "orchestrator", "test_gen": "qwen25_32b", "x": ""},
    )
    conn.close()
    mo = (rec.operator_input or {}).get("model_overrides")
    assert mo == {"code_gen": "orchestrator", "test_gen": "qwen25_32b"}  # empty dropped


def test_model_overrides_from_op_filters_empty():
    assert runner._model_overrides_from_op(
        {"model_overrides": {"a": "b", "c": "", "d": None}}
    ) == {"a": "b"}
    assert runner._model_overrides_from_op({}) == {}
    assert runner._model_overrides_from_op(None) == {}
    assert runner._model_overrides_from_op({"model_overrides": "bad"}) == {}


def test_next_leg_agents():
    assert cp_manager.next_leg_agents("code_gen") == ["code_gen", "test_gen", "code_review"]
    assert cp_manager.next_leg_agents("methodology") == ["methodology"]
    assert cp_manager.next_leg_agents("results_validator") == ["results_validator"]
    assert cp_manager.next_leg_agents("paper_writer") == ["paper_writer", "peer_reviewer"]
    assert cp_manager.next_leg_agents("repo_publisher") == ["repo_publisher"]
    assert cp_manager.next_leg_agents("DONE") == []
    assert cp_manager.next_leg_agents("") == []


def test_validate_model_overrides():
    clean, err = queries.validate_model_overrides(
        {"code_gen": "orchestrator", "test_gen": "qwen25_32b"}
    )
    assert err is None and clean == {"code_gen": "orchestrator", "test_gen": "qwen25_32b"}

    # orchestrator only for orchestratable agents
    clean, err = queries.validate_model_overrides({"code_review": "orchestrator"})
    assert clean == {} and "orchestrator" in err

    # unknown alias rejected
    clean, err = queries.validate_model_overrides({"code_gen": "no_such_model"})
    assert clean == {} and "unknown model" in err

    # empty value = use default (dropped, not an error)
    clean, err = queries.validate_model_overrides({"code_gen": ""})
    assert err is None and clean == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
