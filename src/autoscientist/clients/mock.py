"""Mock provider for smoke testing.

Returns deterministic responses based on agent name. Used by:

  * ``echo`` / ``handoff`` Phase 1 stubs — counter-passing protocol.
  * The 10 real Phase 2 agents — schema-valid JSON fixtures so
    ``smoke_phase2.py`` can exercise the chain without touching Anthropic
    or Ollama.

Each Phase 2 fixture emits the structured JSON its agent's prompt
documents (see ``prompts/<name>.md``) followed by a ``HANDOFF:`` directive
that routes along the canonical chain. The fixtures are minimal — they
preserve the *shape* of agent output so structural assertions and chain
plumbing can be tested. Real agent quality is not exercised here; that is
deferred to Phase 8 end-to-end runs against actual models.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from autoscientist.clients.base import CompletionResult, ToolCall

# ---------------------------------------------------------------------------
# Phase 1 stubs: echo / handoff with COUNT-passing protocol.
# ---------------------------------------------------------------------------

_COUNTER_RE = re.compile(r"COUNT\s+(\d+)")
_DEFAULT_COUNT = 3


def _phase1_echo(inbound: str) -> str:
    m = _COUNTER_RE.search(inbound)
    counter = int(m.group(1)) if m else _DEFAULT_COUNT
    return (
        f"HANDOFF: handoff\n"
        f"COUNT {counter}\n"
        f"echo-saw: {inbound[:80]}"
    )


def _phase1_handoff(inbound: str) -> str:
    m = _COUNTER_RE.search(inbound)
    counter = int(m.group(1)) if m else _DEFAULT_COUNT
    new_count = counter - 1
    if new_count <= 0:
        return "HANDOFF: DONE\nfinished after handoff chain"
    return (
        f"HANDOFF: echo\n"
        f"COUNT {new_count}\n"
        f"handoff-passing"
    )


# ---------------------------------------------------------------------------
# Phase 2 fixtures: schema-valid JSON, canonical handoff chain.
#
# Each fixture takes the raw inbound user-text and returns the agent's
# assistant output. The runner records that output verbatim in messages;
# smoke_phase2 reads it back and asserts schema-key presence.
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object from a string. Returns None on miss."""
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
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def _emit(body: dict[str, Any], handoff_to: str, payload: dict[str, Any] | None = None) -> str:
    """Emit ``<json body>\n\nHANDOFF: <to>\n<json payload>``."""
    parts = [json.dumps(body, indent=2), "", f"HANDOFF: {handoff_to}"]
    if payload is not None:
        parts.append(json.dumps(payload))
    return "\n".join(parts)


def _fix_lit_review(inbound: str) -> str:
    inbound_obj = _try_parse_json(inbound) or {}
    direction = inbound_obj.get("direction", inbound[:120])
    body = {
        "summary": (
            "Mock literature digest for the smoke chain. "
            "In real runs Phase 3 tools fill this in."
        ),
        "key_works": [
            {
                "title": "[CITATION NEEDED] mock work A",
                "authors": ["Anon"],
                "year": 2024,
                "venue": "mock",
                "doi_or_arxiv": "[CITATION NEEDED]",
                "relevance": "high",
                "why": "scaffold for the smoke chain",
            }
        ],
        "gaps": ["mock gap 1", "mock gap 2"],
        "consensus": ["mock consensus claim 1"],
        "disagreements": ["mock disagreement 1"],
        "tools_needed": True,
    }
    payload = {"direction": direction, "lit_digest": body}
    return _emit(body, "idea_gen", payload)


def _fix_idea_gen(inbound: str) -> str:
    inbound_obj = _try_parse_json(inbound) or {}
    direction = inbound_obj.get("direction", "(no direction)")
    lit_digest = inbound_obj.get("lit_digest", {})
    grounding = "weak" if lit_digest.get("tools_needed") else "strong"
    ideas = [
        {
            "title": f"Mock idea {i+1} for: {str(direction)[:40]}",
            "summary": f"Synthetic placeholder idea {i+1}.",
            "literature_gap": f"mock gap {i+1}",
            "novelty": ["med", "med", "high", "low", "high"][i],
            "feasibility": ["high", "high", "med", "high", "low"][i],
            "expected_experiments": [
                f"E{i+1}: train ResNet-50 on mock subset; report AUROC."
            ],
            "compute_estimate": "~2 GPU-hours on a 5090",
            "failure_modes": [f"mock failure mode {i+1}"],
            "grounding": grounding,
        }
        for i in range(5)
    ]
    body = {"ideas": ideas}
    return _emit(body, "idea_critic", {"ideas": ideas})


def _fix_idea_critic(inbound: str) -> str:
    inbound_obj = _try_parse_json(inbound) or {}
    ideas = inbound_obj.get("ideas", [])
    n = len(ideas) if isinstance(ideas, list) else 0
    n = n if n > 0 else 5
    critiques = [
        {
            "idea_index": i,
            "concerns": [f"mock concern {i+1}"],
            "kill_criteria": [f"if mock metric < 0.{50+i}, kill"],
            "potential_confounds": ["scanner shift"],
            "recommendation": "advance" if i == 0 else "revise",
            "rationale": f"Mock rationale {i+1}.",
        }
        for i in range(n)
    ]
    ranked = list(range(n))
    body = {
        "critiques": critiques,
        "ranked_indices": ranked,
        "top_pick": ranked[0],
        "operator_questions": ["Is the mock direction acceptable?"],
    }
    top_idea = ideas[0] if isinstance(ideas, list) and ideas else {"title": "mock"}
    payload = {"top_idea": top_idea, "critique": critiques[0]}
    return _emit(body, "methodology", payload)


def _fix_methodology(inbound: str) -> str:
    inbound_obj = _try_parse_json(inbound) or {}
    top_idea = inbound_obj.get("top_idea", {})
    title = top_idea.get("title", "mock idea")
    plan = {
        "research_question": f"How does mock setting affect {title}?",
        "hypotheses": [
            {"id": "H1", "statement": "mock hypothesis", "predicted_direction": "increase"}
        ],
        "datasets": [
            {
                "name": "NIH ChestX-ray14",
                "role": "train",
                "split_strategy": "patient-level random",
                "fetch_method": "kaggle",
                "preprocessing": ["resize-224", "imagenet-norm"],
            },
            {
                "name": "PadChest",
                "role": "external_val",
                "split_strategy": "patient-level random",
                "fetch_method": "bimcv",
                "preprocessing": ["resize-224", "imagenet-norm"],
            },
        ],
        "baselines": [
            {
                "name": "Rajpurkar CheXNet (mock reference)",
                "expected_metric": "AUROC ~0.84 on NIH pneumonia",
                "tolerance": "+/- 0.02",
            }
        ],
        "metrics": [{"name": "AUROC", "primary": True, "ci_method": "bootstrap n=1000"}],
        "experiments": [
            {
                "id": "E1",
                "describes": "H1",
                "intervention": "training-set size in {1k,5k,25k,100k}",
                "n_seeds": 3,
                "compute_budget": "~20 GPU-hours total on a 5090",
            }
        ],
        "stats_plan": {
            "primary_test": "DeLong test on AUROC differences",
            "alpha": 0.05,
            "multiple_comparisons": "Holm-Bonferroni",
            "effect_size_floor": "AUROC delta >= 0.01",
        },
        "pitfall_acks": [
            {"pitfall": "patient-level (not image-level) split", "mitigation": "split by patient_id"}
        ],
        "stop_conditions": {
            "early_success": "baseline reproduced and external AUROC > 0.80 at N=100k",
            "early_abort": "baseline reproduction fails",
        },
    }
    body = {"plan": plan}
    # Smoke chain terminates here. Real runs continue: HANDOFF: code_gen.
    return _emit(body, "DONE")


def _fix_code_gen(inbound: str) -> str:
    body = {
        "files": [
            {"path": "src/data.py", "content": "# mock data loader\n"},
            {"path": "src/train.py", "content": "# mock train loop\n"},
            {"path": "scripts/run.sh", "content": "#!/usr/bin/env bash\necho mock\n"},
        ],
        "entrypoint": "scripts/run.sh",
        "run_cmd": "bash scripts/run.sh --seed 0 --train-size 1000",
        "dependencies": ["torch", "torchvision", "pandas", "scikit-learn"],
        "notes": "mock fixture for smoke",
    }
    payload = {
        "files": body["files"],
        "entrypoint": body["entrypoint"],
        "run_cmd": body["run_cmd"],
        "plan_step": "mock plan step",
    }
    return _emit(body, "test_gen", payload)


def _fix_test_gen(inbound: str) -> str:
    body = {
        "test_files": [
            {"path": "tests/test_data_split.py", "content": "# mock split test\n"},
            {"path": "tests/test_metrics.py", "content": "# mock metric test\n"},
        ],
        "coverage_targets": [
            "patient-level split uniqueness",
            "AUROC matches sklearn within 1e-6",
            "seed determinism for first epoch",
        ],
        "run_cmd": "pytest tests/ -x -q",
    }
    payload = {
        "src_files": [],
        "test_files": body["test_files"],
        "run_cmd_src": "bash scripts/run.sh",
        "run_cmd_tests": body["run_cmd"],
    }
    return _emit(body, "code_review", payload)


def _fix_code_review(inbound: str) -> str:
    body = {
        "findings": [
            {
                "severity": "minor",
                "file": "src/train.py",
                "lines": "1",
                "issue": "mock placeholder",
                "fix_suggestion": "replace with real implementation",
                "category": "style",
            }
        ],
        "verdict": "pass",
        "summary": "Mock review — no real findings.",
    }
    return _emit(body, "results_validator", {"src_files": [], "test_files": []})


def _fix_results_validator(inbound: str) -> str:
    body = {
        "checks": [
            {"name": "baseline reproduction in tolerance", "status": "pass", "detail": "mock"},
            {"name": "no patient-level leakage", "status": "pass", "detail": "mock"},
            {"name": "seed variance plausible", "status": "pass", "detail": "mock"},
            {"name": "external validation present (if claimed)", "status": "pass", "detail": "mock"},
        ],
        "counterintuitive_findings": [],
        "anomalies": [],
        "verdict": "advance",
        "operator_payload": "Mock results — no concerns.",
    }
    return _emit(body, "paper_writer", {"plan": {}, "results": {}, "validator_summary": body})


def _fix_paper_writer(inbound: str) -> str:
    sections = {
        "title": "Mock paper title",
        "abstract": "Mock abstract.",
        "intro": "Mock intro.",
        "related_work": "Mock related work.",
        "methods": "Mock methods.",
        "results": "Mock results.",
        "discussion": "Mock discussion.",
        "limitations": "Mock limitations.",
        "references": [
            {
                "key": "MockRef2024",
                "title": "[CITATION NEEDED]",
                "authors": ["Anon"],
                "year": 2024,
                "venue": "mock",
                "doi_or_arxiv": "[CITATION NEEDED]",
                "verified": False,
            }
        ],
    }
    body = {
        "sections": sections,
        "supplementary": {
            "datasheet": "Mock datasheet.",
            "model_card": "Mock model card.",
            "extended_results": "Mock extended results.",
        },
        "citation_keys_used": ["MockRef2024"],
    }
    return _emit(body, "peer_reviewer", {"draft": sections, "supplementary": body["supplementary"]})


def _fix_peer_reviewer(inbound: str) -> str:
    body = {
        "review": {
            "summary": "Mock peer review.",
            "strengths": ["mock strength"],
            "weaknesses": [{"severity": "minor", "issue": "mock", "suggested_fix": "mock"}],
            "requested_changes": ["mock change"],
            "missed_pitfalls": [],
        },
        "recommendation": "minor_revise",
        "score": 5,
        "would_re_review": True,
    }
    return _emit(body, "paper_writer", {"prior_draft": {}, "review": body})


# ---------------------------------------------------------------------------
# Phase 6 fixtures — judge + meta_prompter.
#
# The judge fixture decodes a deterministic envelope from the inbound
# payload (the eval harness embeds ``__mock_scores`` for tests so the
# smoke can assert specific outcomes). When that envelope is absent it
# returns a default 3/5 across every dimension named in the rubric so
# the harness can still run end-to-end on a brand-new rubric.
# ---------------------------------------------------------------------------


def _fix_judge(inbound: str) -> str:
    obj = _try_parse_json(inbound) or {}
    rubric_dims = obj.get("rubric_dims") or []
    forced = obj.get("__mock_scores") or {}
    scores: dict[str, dict[str, Any]] = {}
    for dim in rubric_dims:
        if dim in forced:
            scores[dim] = {"score": int(forced[dim]),
                           "rationale": f"mock-forced for {dim}"}
        else:
            scores[dim] = {"score": 3, "rationale": f"mock default for {dim}"}
    body = {
        "agent_name": obj.get("agent_name", "unknown"),
        "anchor_id": obj.get("anchor_id", "unknown"),
        "scores": scores,
        "summary": "Mock judge: deterministic placeholder scoring.",
    }
    # Judge does not hand off — single-shot scoring.
    return json.dumps(body, indent=2)


def _fix_meta_prompter(inbound: str) -> str:
    obj = _try_parse_json(inbound) or {}
    n = int(obj.get("n_variants", 2))
    baseline = obj.get("baseline_prompt", "")[:120]
    variants = [
        {
            "prompt_text": f"# Variant {i+1} (mock)\n\n{baseline}\n\n# Mock change {i+1}",
            "hypothesis": f"Mock hypothesis {i+1}: this variant tightens the output schema.",
        }
        for i in range(n)
    ]
    body = {"variants": variants}
    return json.dumps(body, indent=2)


_FIXTURES: dict[str, Callable[[str], str]] = {
    "echo": _phase1_echo,
    "handoff": _phase1_handoff,
    "lit_review": _fix_lit_review,
    "idea_gen": _fix_idea_gen,
    "idea_critic": _fix_idea_critic,
    "methodology": _fix_methodology,
    "code_gen": _fix_code_gen,
    "test_gen": _fix_test_gen,
    "code_review": _fix_code_review,
    "results_validator": _fix_results_validator,
    "paper_writer": _fix_paper_writer,
    "peer_reviewer": _fix_peer_reviewer,
    "judge": _fix_judge,
    "meta_prompter": _fix_meta_prompter,
}


def _has_tool_result(messages: list[dict[str, Any]]) -> bool:
    """Detect whether any prior message is a tool result.

    Supports both Anthropic shape (``user`` role with ``tool_result`` blocks)
    and OpenAI shape (``tool`` role).
    """
    for m in messages:
        if m.get("role") == "tool":
            return True
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
    return False


def _user_text(messages: list[dict[str, Any]]) -> str:
    """Concatenate user-role text content for fixture parsing.

    Only extracts plain-text content. Tool-result blocks are intentionally
    skipped (so the inbound payload reflects what the *operator* sent).
    """
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text") or ""
                    if t:
                        parts.append(t)
    return "\n".join(parts)


def _make_result(
    *, content: str, model: str, system: str | None, messages: list[dict[str, Any]],
    tool_calls: list[ToolCall] | None = None, raw_blocks: Any = None,
) -> CompletionResult:
    prompt_chars = len(system or "") + sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
        for m in messages
    )
    return CompletionResult(
        content=content,
        model=model,
        provider="mock",
        prompt_tokens=max(1, prompt_chars // 4),
        completion_tokens=max(1, len(content) // 4),
        finish_reason="end_turn" if not tool_calls else "tool_use",
        tool_calls=tool_calls or [],
        raw_content_blocks=raw_blocks,
    )


def complete(
    *,
    agent_name: str,
    model: str,
    system: str | None,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> CompletionResult:
    inbound = _user_text(messages)

    # Tool-use simulation: if the agent has any tools wired AND no tool_result
    # has been delivered yet, emit a single tool_use round. The runner will
    # dispatch the tool, append a tool_result, and call back here — second
    # call sees ``_has_tool_result`` and falls through to the normal fixture.
    tool_names = [t.get("name") or t.get("function", {}).get("name") for t in (tools or [])]
    tool_names = [n for n in tool_names if n]
    if tool_names and not _has_tool_result(messages):
        # Pick a sensible default per agent.
        plan = _MOCK_TOOL_PLAN.get(agent_name)
        if plan is not None and plan["tool"] in tool_names:
            tool_input = plan["build_input"](inbound)
            tc = ToolCall(
                id=f"toolu_mock_{agent_name}_{plan['tool']}",
                name=plan["tool"],
                input=tool_input,
            )
            preamble = plan.get("preamble", "")
            raw_blocks = []
            if preamble:
                raw_blocks.append({"type": "text", "text": preamble})
            raw_blocks.append({
                "type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input,
            })
            return _make_result(
                content=preamble, model=model, system=system, messages=messages,
                tool_calls=[tc], raw_blocks=raw_blocks,
            )

    fixture = _FIXTURES.get(agent_name)
    out = fixture(inbound) if fixture is not None else f"HANDOFF: DONE\nmock fallback for {agent_name}"
    return _make_result(content=out, model=model, system=system, messages=messages,
                        raw_blocks=[{"type": "text", "text": out}])


# ---------------------------------------------------------------------------
# Per-agent mock tool plans. Each entry says: when this agent has tools wired,
# the first call should request this tool with this input. The second call
# (after tool_result) emits the normal fixture.
# ---------------------------------------------------------------------------

def _lit_review_input(inbound: str) -> dict[str, Any]:
    obj = _try_parse_json(inbound) or {}
    direction = obj.get("direction") or inbound[:200]
    return {"query": str(direction)[:200], "limit": 5}


_MOCK_TOOL_PLAN: dict[str, dict[str, Any]] = {
    "lit_review": {
        "tool": "literature_search",
        "preamble": "Searching the literature for grounding.",
        "build_input": _lit_review_input,
    },
}
