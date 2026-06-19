"""Smoke tests for the Starlette operator console (``autoscientist.web``).

Exercises the read endpoints, the checkpoint resolver, and the action
endpoints against a temp DB. The detached ``runner --resume`` subprocess
is stubbed so resolving a checkpoint never launches a real runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from autoscientist.checkpoints import manager as cp_manager
from autoscientist.state.db import open_db, record_message, start_run


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "web.db"
    monkeypatch.setenv("AUTOSCIENTIST_DB_PATH", str(db_path))

    # Import after the env override so any module-level config is harmless;
    # db_path() reads the env var at call time regardless.
    from autoscientist.web import app as web_app

    # Never spawn a real resume subprocess from a test.
    monkeypatch.setattr(web_app, "resume_in_background", lambda run_id: True)

    conn = open_db(db_path)
    run_id = start_run(conn, "demo-project", note="smoke")
    record_message(conn, run_id=run_id, agent_name="lit_review", role="user",
                   content="Survey the literature on X.")
    record_message(conn, run_id=run_id, agent_name="lit_review", role="assistant",
                   content="Found 3 relevant papers.\nHANDOFF: idea_gen",
                   model="claude-x", prompt_tokens=100, completion_tokens=50, cost_usd=0.01)
    record_message(conn, run_id=run_id, agent_name="lit_review", role="tool",
                   content=json.dumps({"name": "literature_search", "duration_ms": 42,
                                       "input": {"query": "X"}, "output": ["a", "b"]}))
    record_message(conn, run_id=run_id, agent_name="idea_critic", role="user",
                   content="Critique these ideas.")
    cp_id = cp_manager.open_checkpoint(
        conn, run_id=run_id, stage=1, from_agent="idea_critic", to_agent="methodology",
        agent_output_raw='{"top_pick": 0}', default_payload="payload-for-methodology",
        parsed={"top_pick": 0, "ranked_indices": [0, 1],
                "critiques": [{"idea_index": 0, "recommendation": "accept", "concerns": ["c1"]}],
                "operator_questions": ["Is X feasible?"]},
    )
    conn.commit()
    conn.close()

    with TestClient(web_app.app) as c:
        c.run_id = run_id  # type: ignore[attr-defined]
        c.cp_id = cp_id    # type: ignore[attr-defined]
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"ok": True}


def test_overview(client: TestClient) -> None:
    data = client.get("/api/overview").json()
    assert "budget" in data and "cap" in data["budget"]
    assert any(r["run_id"] == client.run_id for r in data["runs"])
    assert any(cp["checkpoint_id"] == client.cp_id for cp in data["pending"])
    assert data["active_run_id"] == client.run_id


def test_run_detail_has_stages_and_totals(client: TestClient) -> None:
    d = client.get(f"/api/runs/{client.run_id}").json()
    assert len(d["stages"]) == 5
    assert d["stages"][0]["status"] == "pending"  # stage 1 cp is pending
    assert d["pending_checkpoint"]["checkpoint_id"] == client.cp_id
    assert d["current_stage"] == 1
    assert d["totals"]["messages"] == 4
    assert d["current_agent"] == "idea_critic"
    assert any(a["agent"] == "lit_review" for a in d["agents"])


def test_run_detail_404(client: TestClient) -> None:
    assert client.get("/api/runs/run_nope").status_code == 404


def test_messages_feed_tail_and_increment(client: TestClient) -> None:
    full = client.get(f"/api/runs/{client.run_id}/messages").json()
    assert len(full["events"]) == 4
    roles = [e["role"] for e in full["events"]]
    assert roles == ["user", "assistant", "tool", "handoff"] or "tool" in roles
    # tool row parsed into structured preview
    tool_ev = next(e for e in full["events"] if e["role"] == "tool")
    assert tool_ev["tool"]["name"] == "literature_search"
    # incremental: nothing new after the cursor
    inc = client.get(f"/api/runs/{client.run_id}/messages?after={full['cursor']}").json()
    assert inc["events"] == []


def test_timeline_groups_activations_with_prompt(client: TestClient) -> None:
    acts = client.get(f"/api/runs/{client.run_id}/timeline").json()["activations"]
    lit = next(a for a in acts if a["agent"] == "lit_review")
    assert lit["inbound_prompt"] == "Survey the literature on X."
    assert any(t["name"] == "literature_search" for t in lit["tool_calls"])
    assert lit["output"]  # captured assistant output


def test_checkpoint_detail_and_questions(client: TestClient) -> None:
    cp = client.get(f"/api/checkpoints/{client.cp_id}").json()
    assert cp["stage"] == 1
    assert cp["parsed"]["top_pick"] == 0
    assert cp["status"] == "pending"
    # add a question
    r = client.post(f"/api/checkpoints/{client.cp_id}/questions", json={"content": "Why idea 0?"})
    assert r.json()["ok"] is True
    cp2 = client.get(f"/api/checkpoints/{client.cp_id}").json()
    assert cp2["questions"][0]["content"] == "Why idea 0?"


def test_agent_system_prompt(client: TestClient) -> None:
    r = client.get("/api/agents/lit_review/prompt")
    assert r.status_code == 200
    assert r.json()["agent"] == "lit_review"
    assert len(r.json()["system_prompt"]) > 0
    # path-traversal / unknown guard
    assert client.get("/api/agents/..%2f..%2fsecret/prompt").status_code in (404, 400)
    assert client.get("/api/agents/nonesuch/prompt").status_code == 404


def test_resolve_approve_marks_status_and_resumes(client: TestClient) -> None:
    r = client.post(f"/api/checkpoints/{client.cp_id}/resolve",
                    json={"decision": "approve"})
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "approved"
    assert body["resumed"] is True  # stubbed
    # re-resolve must now fail (no longer pending)
    r2 = client.post(f"/api/checkpoints/{client.cp_id}/resolve", json={"decision": "approve"})
    assert r2.json()["ok"] is False


def test_resolve_rejects_invalid_decision(client: TestClient) -> None:
    r = client.post(f"/api/checkpoints/{client.cp_id}/resolve", json={"decision": "bogus"})
    assert r.status_code == 400


def test_pause_and_cancel(client: TestClient) -> None:
    assert client.post(f"/api/runs/{client.run_id}/pause").json()["ok"] is True
    assert client.post(f"/api/runs/{client.run_id}/cancel-pause").json()["ok"] is True


def test_resolve_rerun_marks_decision_and_resumes(client: TestClient) -> None:
    r = client.post(f"/api/checkpoints/{client.cp_id}/resolve",
                    json={"decision": "rerun", "instructions": "be more rigorous"})
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "modified"   # rerun reuses the modified row status
    assert body["resumed"] is True        # stubbed
    cp = client.get(f"/api/checkpoints/{client.cp_id}").json()
    assert cp["operator_input"]["decision"] == "rerun"
    assert cp["operator_input"]["instructions"] == "be more rigorous"


def test_run_checkpoints_history(client: TestClient) -> None:
    cps = client.get(f"/api/runs/{client.run_id}/checkpoints").json()["checkpoints"]
    assert len(cps) == 1
    assert cps[0]["checkpoint_id"] == client.cp_id
    assert cps[0]["stage"] == 1


def test_projects_listing_and_payload(client: TestClient) -> None:
    data = client.get("/api/projects").json()
    assert data["default_agent"] == "lit_review"
    assert isinstance(data["projects"], list)
    # If any real project carries a kickoff payload, the payload endpoint serves it.
    with_payload = [p for p in data["projects"] if p["has_payload"]]
    if with_payload:
        pid = with_payload[0]["id"]
        pr = client.get(f"/api/projects/{pid}/payload")
        assert pr.status_code == 200
        assert len(pr.json()["payload"]) > 0


def test_start_run_launches_detached(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoscientist.web import app as web_app

    calls = {}
    monkeypatch.setattr(web_app.queries, "project_exists", lambda pid: pid == "demo-proj")
    monkeypatch.setattr(web_app.queries, "project_payload", lambda pid: '{"direction": "x"}')

    def fake_start(*, project, agent, payload):
        calls.update(project=project, agent=agent, payload=payload)
        return True

    monkeypatch.setattr(web_app, "start_in_background", fake_start)

    r = client.post("/api/runs/start", json={"project": "demo-proj"})
    assert r.json()["ok"] is True
    assert calls["project"] == "demo-proj"
    assert calls["agent"] == "lit_review"            # defaulted
    assert calls["payload"] == '{"direction": "x"}'  # defaulted from kickoff payload

    # unknown project → 404
    monkeypatch.setattr(web_app.queries, "project_exists", lambda pid: False)
    assert client.post("/api/runs/start", json={"project": "nope"}).status_code == 404
    # missing project → 400
    assert client.post("/api/runs/start", json={}).status_code == 400
