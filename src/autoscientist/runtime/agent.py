"""Agent definition and system-prompt loading.

An ``Agent`` is the static configuration for a role in the pipeline.
Phase 1 keeps it minimal: name, system-prompt path, and the set of
agents this one is allowed to hand off to.

System prompts live as Markdown files under ``prompts/`` with optional
YAML-style frontmatter. Phase 1 parses only ``temperature`` and
``max_tokens``; Phase 2+ may add ``expected_schema``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    system_prompt_path: Path
    handoff_targets: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    # Keys of MCP servers (see config/mcp.toml) whose tools this agent uses.
    # The runner lazily connects to each and registers its tools before the
    # agent's tool-use loop. Names in ``tools`` that come from an MCP server
    # are the prefixed names (e.g. ``github_push_files``); if the server is
    # unavailable at run time those tools are dropped and the agent proceeds
    # with its native tools only (graceful degradation).
    mcp_servers: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedPrompt:
    system_text: str
    temperature: float | None = None
    max_tokens: int | None = None
    raw_frontmatter: dict[str, Any] = field(default_factory=dict)


def _parse_simple_yaml(s: str) -> dict[str, Any]:
    """Parse ``key: value`` lines. No nesting, no lists. Phase 1 stub.

    Sufficient for ``temperature: 0.7`` / ``max_tokens: 4096`` / ``model: foo``.
    """
    out: dict[str, Any] = {}
    for raw_line in s.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            val = val[1:-1]
        if val.lower() in {"true", "false"}:
            out[key] = val.lower() == "true"
            continue
        try:
            out[key] = float(val) if "." in val else int(val)
            continue
        except ValueError:
            pass
        out[key] = val
    return out


def load_prompt(path: Path) -> LoadedPrompt:
    text = path.read_text(encoding="utf-8")
    fm: dict[str, Any] = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        fm = _parse_simple_yaml(m.group(1))
        body = text[m.end():]
    temp = fm.get("temperature")
    mt = fm.get("max_tokens")
    return LoadedPrompt(
        system_text=body.strip(),
        temperature=float(temp) if temp is not None else None,
        max_tokens=int(mt) if mt is not None else None,
        raw_frontmatter=fm,
    )
