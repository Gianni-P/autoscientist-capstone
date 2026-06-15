"""Phase 3 tool integrations.

Each tool lives in its own module:

  * ``literature`` — Semantic Scholar / OpenAlex / arxiv lookup.
  * ``pdf_parse`` — pypdf-based extraction with sha256 caching.
  * ``execute`` — sandboxed subprocess runner with resource limits.
  * ``datasets`` — public dataset registry + fetchers (Kaggle, BIMCV).
  * ``latex`` — tectonic-based LaTeX → PDF compilation.
  * ``citation_check`` — round-trip verification of cited works.
  * ``write_file`` — write a file into the project sandbox.

Tools are pure functions (or thin classes) that callers invoke directly.
Phase 3.5 wires these into the agent runtime via Anthropic tool-use; this
module provides the implementations only.

Tool result caching uses the ``tool_cache`` table (see ``state/db.py``) — a
generic per-tool key/value store keyed on a SHA256 of canonicalized inputs.
"""

from __future__ import annotations

from autoscientist.tools.tool_cache import cache_get, cache_key, cache_put

__all__ = ["cache_get", "cache_key", "cache_put"]
