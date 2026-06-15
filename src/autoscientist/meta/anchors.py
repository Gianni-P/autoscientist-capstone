"""Anchor example loader.

KICKOFF.md §9 Phase 6: "Curate 10-30 anchor examples per agent
(gold-standard outputs)."

Each anchor is a JSON file under ``prompts/anchors/<agent_name>/`` with
the schema:

    {
      "anchor_id": "idea_gen_01_strong",
      "agent": "idea_gen",
      "input_payload": "<JSON or free-text the agent will see as user msg>",
      "expected_summary": "<free-text description of what good output looks like>",
      "expected_keys": ["ideas[].title", "ideas[].grounding"],
      "notes": "..."
    }

``input_payload`` is whatever string the agent receives as its inbound
user message. ``expected_summary`` is fed to the judge so the rubric has
a target to compare against. ``expected_keys`` is optional — when set,
the harness can do a cheap structural pre-check before LLM judging.

Anchors live as files (not in SQLite) so they version-control with the
prompts and stay reviewable by the operator outside any tooling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    agent: str
    input_payload: str
    expected_summary: str
    expected_keys: tuple[str, ...] = ()
    notes: str = ""
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "anchor_id": self.anchor_id,
            "agent": self.agent,
            "input_payload": self.input_payload,
            "expected_summary": self.expected_summary,
            "expected_keys": list(self.expected_keys),
            "notes": self.notes,
        }
        if self.source_path is not None:
            d["source_path"] = self.source_path
        return d


@dataclass(frozen=True)
class AnchorSet:
    agent: str
    anchors: tuple[Anchor, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.anchors)

    def __iter__(self):
        return iter(self.anchors)

    def by_id(self, anchor_id: str) -> Anchor | None:
        for a in self.anchors:
            if a.anchor_id == anchor_id:
                return a
        return None


def anchors_dir(prompts_dir: Path, agent: str) -> Path:
    return prompts_dir / "anchors" / agent


def load_anchor_file(path: Path) -> Anchor:
    raw = json.loads(path.read_text(encoding="utf-8"))
    aid = raw.get("anchor_id")
    agent = raw.get("agent")
    inp = raw.get("input_payload")
    exp = raw.get("expected_summary")
    if not aid or not agent or inp is None or exp is None:
        raise ValueError(
            f"anchor {path} missing required fields "
            "(anchor_id, agent, input_payload, expected_summary)"
        )
    keys_raw = raw.get("expected_keys") or []
    if not isinstance(keys_raw, list):
        raise ValueError(f"anchor {path}: expected_keys must be a list")
    return Anchor(
        anchor_id=str(aid),
        agent=str(agent),
        input_payload=str(inp),
        expected_summary=str(exp),
        expected_keys=tuple(str(k) for k in keys_raw),
        notes=str(raw.get("notes") or ""),
        source_path=str(path),
    )


def load_anchor_set(
    prompts_dir: Path, agent: str, *, strict: bool = True,
) -> AnchorSet:
    """Load every ``*.json`` file under ``prompts/anchors/<agent>/``.

    Files are sorted by name for determinism. ``strict=True`` (default)
    raises if a file fails to parse; ``strict=False`` skips the bad file
    after logging — useful while curating new anchors that aren't valid
    yet.
    """
    base = anchors_dir(prompts_dir, agent)
    if not base.exists():
        return AnchorSet(agent=agent, anchors=())
    found: list[Anchor] = []
    seen_ids: set[str] = set()
    for path in sorted(base.glob("*.json")):
        try:
            a = load_anchor_file(path)
        except (ValueError, json.JSONDecodeError):
            if strict:
                raise
            continue
        if a.agent != agent:
            raise ValueError(
                f"anchor {path}: declared agent {a.agent!r} does not "
                f"match directory {agent!r}"
            )
        if a.anchor_id in seen_ids:
            raise ValueError(f"duplicate anchor_id {a.anchor_id!r} in {path}")
        seen_ids.add(a.anchor_id)
        found.append(a)
    return AnchorSet(agent=agent, anchors=tuple(found))


def write_anchor(
    prompts_dir: Path, anchor: Anchor, *, overwrite: bool = False,
) -> Path:
    """Persist an anchor to its canonical location. Returns the path."""
    out = anchors_dir(prompts_dir, anchor.agent) / f"{anchor.anchor_id}.json"
    if out.exists() and not overwrite:
        raise FileExistsError(f"anchor file already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anchor_id": anchor.anchor_id,
        "agent": anchor.agent,
        "input_payload": anchor.input_payload,
        "expected_summary": anchor.expected_summary,
        "expected_keys": list(anchor.expected_keys),
        "notes": anchor.notes,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def has_expected_keys(anchor: Anchor, parsed_output: dict[str, Any]) -> tuple[bool, list[str]]:
    """Cheap pre-judge structural check.

    ``expected_keys`` uses a dotted path with ``[]`` for "any element of
    a list" — e.g. ``ideas[].title`` requires ``parsed_output["ideas"]``
    to be a non-empty list whose elements each contain a ``title`` key.

    Returns ``(ok, missing_paths)``.
    """
    missing: list[str] = []
    for path in anchor.expected_keys:
        if not _key_present(parsed_output, path):
            missing.append(path)
    return (not missing, missing)


def _key_present(obj: Any, path: str) -> bool:
    return _walk(obj, path.split("."))


def _walk(cur: Any, parts: list[str]) -> bool:
    for i, p in enumerate(parts):
        if p.endswith("[]"):
            head = p[:-2]
            if not isinstance(cur, dict) or head not in cur:
                return False
            inner = cur[head]
            if not isinstance(inner, list) or not inner:
                return False
            tail = parts[i + 1 :]
            if not tail:
                return True
            return all(_walk(elem, tail) for elem in inner)
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True
