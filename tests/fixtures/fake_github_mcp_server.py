"""A tiny in-memory MCP server that imitates the GitHub MCP server's publish
surface — used to test the bridge + registry integration offline (no Docker,
no network, no real GitHub).

It exposes the small subset of *write* tools ``repo_publisher`` actually uses,
with the same names and roughly the same input shapes as the official
``github/github-mcp-server`` (verified June 2026) so the integration test
exercises the real call path:

  * ``get_me``               (users toolset)
  * ``create_repository``    (repos)  — ``autoInit`` creates the default branch
  * ``create_branch``        (repos)
  * ``push_files``           (repos)  — multi-file single commit; branch must exist
  * ``create_or_update_file``(repos)  — single file

There is deliberately NO ``create_release`` tool: the real server has none
(release/tag tools are read-only), and the integration test asserts that
publishing works without it. State lives in memory and resets each process.

Run standalone over stdio (how the bridge launches it in tests)::

    python tests/fixtures/fake_github_mcp_server.py
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fake-github")

# In-memory state so a test can assert the server "did" something.
_REPOS: dict[str, dict[str, Any]] = {}
_FILES: dict[str, dict[str, str]] = {}
_BRANCHES: dict[str, set[str]] = {}

_OWNER = "test-operator"


@mcp.tool(description="Get details of the authenticated GitHub user.")
def get_me() -> dict[str, Any]:
    return {"login": _OWNER, "type": "User", "name": "Test Operator"}


@mcp.tool(description="Create a new GitHub repository for the authenticated user.")
def create_repository(
    name: str,
    description: str = "",
    private: bool = False,
    autoInit: bool = False,  # noqa: N803 — mirrors the real GitHub tool param name
) -> dict[str, Any]:
    full = f"{_OWNER}/{name}"
    _REPOS[full] = {
        "full_name": full,
        "description": description,
        "private": private,
        "default_branch": "main",
        "html_url": f"https://github.com/{full}",
    }
    _FILES.setdefault(full, {})
    # autoInit creates an initial commit on the default branch so push_files,
    # which requires an existing branch, can target "main" immediately.
    _BRANCHES[full] = {"main"} if autoInit else set()
    return _REPOS[full]


@mcp.tool(description="Create a new branch from an existing one.")
def create_branch(
    owner: str, repo: str, branch: str, from_branch: str = "main"
) -> dict[str, Any]:
    full = f"{owner}/{repo}"
    _BRANCHES.setdefault(full, set()).add(branch)
    return {"ref": f"refs/heads/{branch}", "from": from_branch}


@mcp.tool(description="Push several files to a GitHub repository in a single commit.")
def push_files(
    owner: str,
    repo: str,
    branch: str,
    message: str,
    files: list[dict[str, str]],
) -> dict[str, Any]:
    full = f"{owner}/{repo}"
    if branch not in _BRANCHES.get(full, set()):
        # Mirror the real server: the target branch must already exist.
        raise ValueError(
            f"branch {branch!r} does not exist in {full}; "
            f"create the repo with autoInit or call create_branch first"
        )
    store = _FILES.setdefault(full, {})
    for f in files:
        store[f["path"]] = f.get("content", "")
    return {
        "commit": {"message": message, "branch": branch},
        "files_pushed": len(files),
        "html_url": f"https://github.com/{full}/tree/{branch}",
    }


@mcp.tool(description="Create or update a single file in a repository.")
def create_or_update_file(
    owner: str,
    repo: str,
    path: str,
    content: str,
    message: str,
    branch: str,
    sha: str = "",
) -> dict[str, Any]:
    full = f"{owner}/{repo}"
    _FILES.setdefault(full, {})[path] = content
    return {
        "commit": {"message": message, "branch": branch},
        "content": {"path": path},
        "html_url": f"https://github.com/{full}/blob/{branch}/{path}",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
