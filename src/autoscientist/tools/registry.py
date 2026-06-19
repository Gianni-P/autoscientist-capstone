"""Central tool registry for the Phase 3.5 LLM tool-use loop.

A :class:`ToolSpec` packages a tool's name, description, JSON-schema input
contract, and Python handler. The registry exposes:

  * ``get_specs(names)`` — resolve a tuple of tool names to specs.
  * ``anthropic_schemas(specs)`` — list of dicts in Anthropic ``tools=`` shape.
  * ``openai_schemas(specs)``    — list of dicts in OpenAI tool-calling shape.
  * ``dispatch(name, input, ctx)`` — run the handler and return a JSON-safe dict.

The runner (``runtime/runner.py``) calls these. Per agent, only the tools
declared in ``Agent.tools`` are exposed to the LLM. Other tools are *not*
visible to that agent — capability restriction at the schema layer is the
whole point.

Handlers receive a :class:`ToolContext` carrying the SQLite connection,
project root, and config so they can use the same caches the direct
function calls would.

Adding a new tool: write the handler in its module (e.g. ``literature.py``),
then call ``register(...)`` here at module import time. Keep input_schema
strict — Anthropic enforces it, but more importantly the LLM uses the
schema as a description.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("autoscientist.tools.registry")


@dataclass
class ToolContext:
    """Runtime context passed to every tool handler.

    Tools that don't need a piece (e.g. ``literature_search`` doesn't need
    ``project_id``) can ignore the field.
    """
    conn: sqlite3.Connection | None = None
    project_id: str | None = None
    projects_root: Path | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any], ToolContext], Any]


_REGISTRY: dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"tool already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def register_or_replace(spec: ToolSpec) -> None:
    """Register a tool, overwriting any existing spec of the same name.

    Used for *dynamically discovered* tools (e.g. MCP server tools, which are
    enumerated from a live server and re-registered each time the server is
    (re)connected). Static native tools use :func:`register`, which refuses
    duplicates so a genuine double-registration bug surfaces loudly.
    """
    _REGISTRY[spec.name] = spec


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def unregister(name: str) -> None:
    """Remove a tool from the registry if present. Used to retract dynamically
    registered tools (e.g. MCP server tools) when their server disconnects, so
    a dead tool is never offered to an agent."""
    _REGISTRY.pop(name, None)


def get_spec(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown tool: {name}")
    return _REGISTRY[name]


def get_specs(names: tuple[str, ...] | list[str]) -> list[ToolSpec]:
    return [get_spec(n) for n in names]


def anthropic_schemas(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Render specs into Anthropic ``tools=`` shape:

        {"name": str, "description": str, "input_schema": {...}}
    """
    return [
        {
            "name": s.name,
            "description": s.description,
            "input_schema": s.input_schema,
        }
        for s in specs
    ]


def openai_schemas(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Render specs into OpenAI tool-calling shape:

        {"type": "function",
         "function": {"name": str, "description": str, "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.input_schema,
            },
        }
        for s in specs
    ]


def tools_signature(specs: list[ToolSpec]) -> str:
    """Stable signature string for the cache key.

    The signature captures the schema set, so two requests with the same
    tools-set hit the same cache entry.
    """
    payload = sorted(
        [(s.name, s.description, s.input_schema) for s in specs], key=lambda t: t[0]
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass
class DispatchResult:
    name: str
    input: dict[str, Any]
    output: Any
    error: str | None
    duration_ms: int


def dispatch(name: str, input_dict: dict[str, Any], ctx: ToolContext) -> DispatchResult:
    """Run ``name`` with ``input_dict`` against ``ctx``. Never raises — errors
    are captured into ``DispatchResult.error`` so the loop can still feed a
    ``tool_result`` back to the LLM (otherwise the conversation gets stuck).
    """
    started = time.monotonic()
    try:
        spec = get_spec(name)
    except KeyError as e:
        return DispatchResult(
            name=name, input=input_dict, output=None,
            error=f"unknown_tool:{e}", duration_ms=0,
        )
    try:
        out = spec.handler(input_dict, ctx)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "tools.dispatch.ok",
            name=name, duration_ms=elapsed_ms,
            run_id=ctx.run_id, project_id=ctx.project_id,
        )
        return DispatchResult(
            name=name, input=input_dict, output=out,
            error=None, duration_ms=elapsed_ms,
        )
    except Exception as e:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.warning(
            "tools.dispatch.error",
            name=name, error=str(e), error_type=type(e).__name__,
            duration_ms=elapsed_ms,
        )
        return DispatchResult(
            name=name, input=input_dict, output=None,
            error=f"{type(e).__name__}: {e}", duration_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Tool definitions. Each handler is a thin adapter that:
#   1. Validates/coerces inputs (the LLM may pass extras).
#   2. Calls into the underlying tool module.
#   3. Returns a JSON-safe dict (dataclasses converted via to_dict()).
# ---------------------------------------------------------------------------

def _h_literature_search(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import literature

    query = str(inp.get("query", "")).strip()
    if not query:
        raise ValueError("literature_search requires a non-empty query")
    try:
        limit = int(inp.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 25))  # bound model-supplied limit
    include_arxiv = bool(inp.get("include_arxiv", False))
    papers = literature.search(
        query, limit=limit, include_arxiv=include_arxiv, conn=ctx.conn,
    )
    return {"results": [p.to_dict() for p in papers], "count": len(papers)}


def _h_literature_lookup(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import literature

    p = literature.lookup(
        doi=inp.get("doi") or None,
        arxiv_id=inp.get("arxiv_id") or None,
        semantic_scholar_id=inp.get("semantic_scholar_id") or None,
        conn=ctx.conn,
    )
    return {"paper": p.to_dict() if p else None}


def _h_pdf_parse(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import pdf_parse

    path = inp.get("path")
    if not path:
        raise ValueError("pdf_parse requires 'path'")
    doc = pdf_parse.parse_pdf(path, conn=ctx.conn)
    # Cap text payload sent back to the model — long PDFs blow up tokens.
    text = doc.text
    truncated = len(text) > 30_000
    if truncated:
        text = text[:30_000] + "\n…[truncated]"
    return {
        "sha256": doc.sha256,
        "page_count": doc.page_count,
        "text": text,
        "text_truncated": truncated,
        "metadata": doc.metadata,
    }


def _h_execute(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import execute as ex

    if ctx.project_id is None or ctx.projects_root is None:
        raise ValueError("execute tool requires project_id and projects_root in context")
    cmd = inp.get("cmd")
    if not cmd or not isinstance(cmd, list):
        raise ValueError("execute requires 'cmd' as a non-empty list of strings")

    def _capped(key: str, default: int) -> int:
        """Model may only TIGHTEN a resource limit, never loosen it.

        A malformed or out-of-range value (incl. 0/neg, which would mean
        'no cap') falls back to the configured default ceiling rather than
        crashing the tool call or removing the limit.
        """
        raw = inp.get(key)
        if raw is None:
            return default
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return default
        return min(n, default) if n > 0 else default

    res = ex.execute(
        cmd,
        project_id=ctx.project_id,
        projects_root=ctx.projects_root,
        timeout_seconds=_capped("timeout_seconds", ex.DEFAULT_TIMEOUT_SECONDS),
        cpu_seconds=_capped("cpu_seconds", ex.DEFAULT_CPU_SECONDS),
        memory_bytes=_capped("memory_bytes", ex.DEFAULT_MEMORY_BYTES),
    )
    # Truncate stdout/stderr returned to the model.
    out = res.to_dict()
    for k in ("stdout", "stderr"):
        v = out.get(k) or ""
        if len(v) > 8000:
            out[k] = v[:8000] + "\n…[truncated]"
    return out


def _h_dataset_info(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import datasets

    name = inp.get("name")
    if name:
        if name not in datasets.DATASET_REGISTRY:
            raise KeyError(f"unknown dataset: {name}")
        spec = datasets.DATASET_REGISTRY[name]
        return {"name": name, "spec": {
            "source": spec.source, "description": spec.description,
            "license_note": spec.license_note, "citation": spec.citation,
            "kaggle_dataset": spec.kaggle_dataset, "expected_files": spec.expected_files,
        }}
    return {
        "registry": [
            {"name": n, "source": s.source, "description": s.description}
            for n, s in datasets.DATASET_REGISTRY.items()
        ]
    }


def _h_dataset_fetch(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import datasets

    name = inp.get("name")
    if not name:
        raise ValueError("dataset_fetch requires 'name'")
    dest = inp.get("dest_dir")
    if not dest:
        if ctx.projects_root is None or ctx.project_id is None:
            raise ValueError("dataset_fetch needs dest_dir or project context")
        dest = ctx.projects_root / ctx.project_id / "datasets" / name
    try:
        path = datasets.fetch_dataset(name, dest_dir=dest, bimcv_url=inp.get("bimcv_url"))
        return {"fetched": True, "dir": str(path)}
    except datasets.FetchSkippedError as e:
        return {"fetched": False, "skipped_reason": str(e)}


def _h_latex_compile(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import latex

    tex = inp.get("tex_source")
    if not tex:
        raise ValueError("latex_compile requires 'tex_source'")
    if ctx.projects_root is None or ctx.project_id is None:
        raise ValueError("latex_compile needs project context")
    out_dir = ctx.projects_root / ctx.project_id / "latex" / inp.get("job_name", "paper")
    if not latex.is_available():
        return {"success": False, "error": "tectonic not on PATH"}
    build = latex.compile_latex(
        tex, output_dir=out_dir, job_name=inp.get("job_name", "paper"), conn=ctx.conn,
    )
    return {
        "success": build.success,
        "pdf_path": build.pdf_path,
        "errors": build.errors[:20],
        "warnings": build.warnings[:20],
        "duration_ms": build.duration_ms,
    }


def _h_citation_check(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import citation_check

    citation = inp.get("citation")
    if not citation:
        raise ValueError("citation_check requires 'citation' object")
    chk = citation_check.verify_citation(citation, conn=ctx.conn)
    return chk.to_dict()


def _h_write_file(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import write_file as wf

    path = inp.get("path")
    content = inp.get("content")
    if not path or not isinstance(path, str):
        raise ValueError("write_file requires 'path' (non-empty string)")
    if content is None:
        raise ValueError("write_file requires 'content'")
    if ctx.project_id is None or ctx.projects_root is None:
        raise ValueError("write_file requires project_id and projects_root in context")
    return wf.write_file(
        path=path,
        content=str(content),
        project_id=ctx.project_id,
        projects_root=ctx.projects_root,
    )


def _h_read_sandbox_file(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import read_sandbox_file as rsf

    path = inp.get("path")
    if not path or not isinstance(path, str):
        raise ValueError("read_sandbox_file requires 'path' (non-empty string)")
    if ctx.project_id is None or ctx.projects_root is None:
        raise ValueError("read_sandbox_file requires project_id and projects_root in context")
    return rsf.read_sandbox_file(
        path=path,
        project_id=ctx.project_id,
        projects_root=ctx.projects_root,
    )


def _h_list_sandbox(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import list_sandbox as ls

    if ctx.project_id is None or ctx.projects_root is None:
        raise ValueError("list_sandbox requires project_id and projects_root in context")
    subdir = inp.get("subdir") or ""
    max_entries = int(inp.get("max_entries", ls.DEFAULT_MAX_ENTRIES))
    return ls.list_sandbox(
        project_id=ctx.project_id,
        projects_root=ctx.projects_root,
        subdir=str(subdir),
        max_entries=max_entries,
    )


def _h_write_release_file(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import write_release_file as wrf

    path = inp.get("path")
    content = inp.get("content")
    if not path or not isinstance(path, str):
        raise ValueError("write_release_file requires 'path' (non-empty string)")
    if content is None:
        raise ValueError("write_release_file requires 'content'")
    if ctx.project_id is None or ctx.projects_root is None:
        raise ValueError("write_release_file requires project_id and projects_root in context")
    return wrf.write_release_file(
        path=path,
        content=str(content),
        project_id=ctx.project_id,
        projects_root=ctx.projects_root,
    )


def _h_check_imports(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from autoscientist.tools import check_imports as ci

    if ctx.project_id is None or ctx.projects_root is None:
        raise ValueError("check_imports requires project_id and projects_root in context")
    return ci.check_imports(
        project_id=ctx.project_id,
        projects_root=ctx.projects_root,
        subdir=str(inp.get("subdir") or ""),
    )


def _h_handoff(inp: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # The runner (runtime/runner._invoke_agent) intercepts `handoff` tool calls
    # to validate `target` against the agent's allowed handoff targets and
    # synthesize the canonical `HANDOFF:` directive that the rest of the pipeline
    # parses. This echo handler only runs if the tool is ever dispatched outside
    # that path; it just records the requested routing so the transcript is sane.
    return {
        "target": str(inp.get("target", "")).strip(),
        "summary": str(inp.get("summary", "") or ""),
        "acknowledged": True,
    }


# ---------------------------------------------------------------------------
# Register defaults. Idempotent guard for repeated imports under pytest etc.
# ---------------------------------------------------------------------------

def _register_defaults() -> None:
    if "literature_search" in _REGISTRY:
        return  # already registered

    register(ToolSpec(
        name="literature_search",
        description=(
            "Search academic literature via Semantic Scholar (primary), OpenAlex "
            "(fallback), and optionally arxiv. Returns a list of papers with "
            "title, authors, year, venue, DOI/arxiv id, abstract, citation count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Free-text search query"},
                "limit": {"type": "integer", "description": "Max results (default 10)", "default": 10},
                "include_arxiv": {"type": "boolean", "description": "Include arxiv preprints", "default": False},
            },
            "required": ["query"],
        },
        handler=_h_literature_search,
    ))

    register(ToolSpec(
        name="literature_lookup",
        description=(
            "Resolve a single paper by DOI, arxiv id, or Semantic Scholar id. "
            "Returns null if the paper cannot be found."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "doi": {"type": "string"},
                "arxiv_id": {"type": "string"},
                "semantic_scholar_id": {"type": "string"},
            },
        },
        handler=_h_literature_lookup,
    ))

    register(ToolSpec(
        name="pdf_parse",
        description=(
            "Extract text from a local PDF file. Returns text (truncated past "
            "30k chars), page count, and metadata."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or project-relative path"}},
            "required": ["path"],
        },
        handler=_h_pdf_parse,
    ))

    register(ToolSpec(
        name="execute",
        description=(
            "Run a program in the project sandbox. NOT a shell: pass an argv "
            "list whose first element is one of python / python3 / pytest "
            "(other executables and shell strings are refused). Outbound network "
            "is blocked. Resource-limited (CPU, memory) and timeout-bounded. "
            "stdout/stderr are truncated past 8k chars for the response. To run "
            "anything else, write a Python script and execute that."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "array", "items": {"type": "string"},
                    "description": (
                        "Argv list (NOT a shell string); cmd[0] must be "
                        "python/python3/pytest. Example: "
                        "['python', 'train.py', '--seed', '0']"
                    ),
                },
                "timeout_seconds": {"type": "integer", "default": 1800},
                "cpu_seconds": {"type": "integer"},
                "memory_bytes": {"type": "integer"},
            },
            "required": ["cmd"],
        },
        handler=_h_execute,
    ))

    register(ToolSpec(
        name="dataset_info",
        description=(
            "Describe a public dataset from the autoscientist registry. "
            "Pass 'name' for one dataset, or omit for the full registry."
        ),
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
        handler=_h_dataset_info,
    ))

    register(ToolSpec(
        name="dataset_fetch",
        description=(
            "Fetch a registered dataset to disk. Idempotent. May skip with "
            "'fetched: false' if credentials are missing (Kaggle/BIMCV)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "dest_dir": {"type": "string"},
                "bimcv_url": {"type": "string"},
            },
            "required": ["name"],
        },
        handler=_h_dataset_fetch,
    ))

    register(ToolSpec(
        name="latex_compile",
        description=(
            "Compile LaTeX source to PDF using tectonic. Returns success flag, "
            "pdf_path, and parsed errors/warnings."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tex_source": {"type": "string"},
                "job_name": {"type": "string", "default": "paper"},
            },
            "required": ["tex_source"],
        },
        handler=_h_latex_compile,
    ))

    register(ToolSpec(
        name="citation_check",
        description=(
            "Verify a single citation by round-tripping through the literature "
            "APIs. Returns verified flag, matched paper, and mismatch reasons."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "citation": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year": {"type": "integer"},
                        "doi_or_arxiv": {"type": "string"},
                    },
                },
            },
            "required": ["citation"],
        },
        handler=_h_citation_check,
    ))

    register(ToolSpec(
        name="write_file",
        description=(
            "Write a file to the project sandbox. Path must be relative to "
            "the sandbox root (e.g. 'src/data.py'). Parent directories are "
            "created automatically. Use this to persist code files one at a "
            "time instead of emitting them all in a single JSON blob."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path inside the sandbox (e.g. 'src/data.py')",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
        handler=_h_write_file,
    ))

    register(ToolSpec(
        name="read_sandbox_file",
        description=(
            "Read a single file from the project sandbox. Returns the text "
            "content (truncated past 30k chars) plus size in bytes. Binary "
            "files surface as encoding='binary' with empty content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path inside the sandbox (e.g. 'src/data.py')",
                },
            },
            "required": ["path"],
        },
        handler=_h_read_sandbox_file,
    ))

    register(ToolSpec(
        name="list_sandbox",
        description=(
            "List files in the project sandbox or one of its subdirectories. "
            "Returns relative POSIX paths and file sizes. Hidden / cache dirs "
            "(__pycache__, .venv, .git, .pytest_cache) are filtered out."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "subdir": {
                    "type": "string",
                    "description": "Optional relative subdir to scope the walk (default: sandbox root).",
                    "default": "",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Cap on entries returned (default 500). Result is marked truncated if hit.",
                    "default": 500,
                },
            },
        },
        handler=_h_list_sandbox,
    ))

    register(ToolSpec(
        name="write_release_file",
        description=(
            "Write a file into the project release directory at "
            "projects/<project_id>/release/. Use this (not write_file) when "
            "producing the curated publishable repository. Parent dirs are "
            "created automatically."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path inside the release dir (e.g. 'README.md' or 'src/data.py')",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
        handler=_h_write_release_file,
    ))

    register(ToolSpec(
        name="check_imports",
        description=(
            "Statically check that every intra-project import in the sandbox "
            "resolves — WITHOUT running any code. Parses all .py files and reports "
            "imports of names no sibling module defines (the #1 cause of rejected "
            "reviews, e.g. `from src.config import TERRAINS` where config.py never "
            "defines TERRAINS). Returns `ok`, `files_checked`, and an `unresolved` "
            "list where each entry names the missing symbol AND what IS available "
            "in that module, so you can fix the import or add the real definition. "
            "Call this before handing off and resolve every entry. Read-only; "
            "nothing is imported or executed, so it cannot run your experiment."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "subdir": {
                    "type": "string",
                    "description": "Optional sandbox-relative subdir to scope the check (default: whole sandbox).",
                    "default": "",
                },
            },
        },
        handler=_h_check_imports,
    ))

    register(ToolSpec(
        name="handoff",
        description=(
            "Finish your turn and route to the next agent. Call this INSTEAD of "
            "writing a 'HANDOFF:' line in prose — it is the reliable way to hand "
            "off and the only one guaranteed to be parsed. `target` must be one of "
            "your allowed handoff targets (or 'DONE' to end the run); an unknown "
            "target is rejected and you may simply call handoff again. Put the "
            "metadata the next agent needs (e.g. files_written, run_cmd, plan_step) "
            "in `summary`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Next agent to run, e.g. 'test_gen' or 'code_review', or 'DONE'.",
                },
                "summary": {
                    "type": "string",
                    "description": "Handoff payload / note for the next agent (JSON object or prose).",
                },
            },
            "required": ["target"],
        },
        handler=_h_handoff,
    ))


_register_defaults()


# Names of all registered tools (for sanity check / introspection).
ALL_TOOL_NAMES: tuple[str, ...] = tuple(sorted(_REGISTRY.keys()))
