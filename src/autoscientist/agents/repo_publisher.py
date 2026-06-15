"""Repo publisher agent — packages the approved sandbox into a release and
publishes it to GitHub.

Terminal node in the agent graph. Reads from the sandbox, writes a curated,
publishable repository to ``projects/<project_id>/release/``, then publishes
that tree to a real GitHub repository via the GitHub MCP server, and emits
``HANDOFF: DONE``. Triggered by peer_reviewer when the recommendation is
``accept`` (after CP5 operator approval).

The ``github_*`` tools are provided by the GitHub MCP server (see
``config/mcp.toml`` and ``tools/mcp_integration.py``). They are best-effort: if
the server is unreachable at run time (no ``GITHUB_PERSONAL_ACCESS_TOKEN``,
Docker/network down) the runner drops them and the agent still writes the local
release tree — the primary deliverable. The GitHub MCP server has no
release/tag tool, so publishing stops at "repo created + files pushed"; tagging
a release is a documented manual step.
"""

from __future__ import annotations

from pathlib import Path

from autoscientist.runtime.agent import Agent

AGENT = Agent(
    name="repo_publisher",
    role="curate publishable repository from approved sandbox and publish to GitHub",
    system_prompt_path=Path("repo_publisher.md"),
    handoff_targets=(),
    tools=(
        # Native tools — build the curated release tree on local disk.
        "list_sandbox",
        "read_sandbox_file",
        "write_release_file",
        "citation_check",
        # GitHub MCP tools — publish the release tree to a real repository.
        # (No get_me: the remote server doesn't expose it; the owner comes from
        # create_repository's response.)
        "github_create_repository",
        "github_push_files",
        "github_create_or_update_file",
    ),
    mcp_servers=("github",),
)
