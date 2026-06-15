"""Centralized config loader for autoscientist.

Reads ``config/default.toml`` and ``config/models.toml`` once per process,
optionally loading a ``.env`` file if present. Resolves all paths relative
to the repository root (the directory containing ``pyproject.toml``).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


_ROOT = project_root()


@dataclass
class Config:
    default: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] = field(default_factory=dict)
    root: Path = field(default_factory=Path)

    def db_path(self) -> Path:
        override = os.environ.get("AUTOSCIENTIST_DB_PATH")
        if override:
            return Path(override)
        rel = self.default.get("paths", {}).get("db_path", "autoscientist.db")
        path = Path(rel)
        return path if path.is_absolute() else (self.root / path)

    def runs_dir(self) -> Path:
        rel = self.default.get("paths", {}).get("runs_dir", "runs")
        return self.root / rel

    def prompts_dir(self) -> Path:
        rel = self.default.get("paths", {}).get("prompts_dir", "prompts")
        return self.root / rel


_config: Config | None = None


def load_config(reload: bool = False) -> Config:
    global _config
    if _config is not None and not reload:
        return _config
    env_file = _ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
    with (_ROOT / "config" / "default.toml").open("rb") as f:
        default = tomllib.load(f)
    with (_ROOT / "config" / "models.toml").open("rb") as f:
        models = tomllib.load(f)
    # MCP server config is optional — the pipeline runs fine without it; only
    # agents that declare `mcp_servers` need it (e.g. repo_publisher → github).
    mcp: dict[str, Any] = {}
    mcp_path = _ROOT / "config" / "mcp.toml"
    if mcp_path.exists():
        with mcp_path.open("rb") as f:
            mcp = tomllib.load(f)
    _config = Config(default=default, models=models, mcp=mcp, root=_ROOT)
    return _config


def reset_for_tests() -> None:
    global _config
    _config = None
