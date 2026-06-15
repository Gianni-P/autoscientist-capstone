"""Spend tracking with a hard monthly cap and optional per-project soft cap.

Budget enforcement is non-negotiable per KICKOFF.md §2:
  * monthly cap defaults to $150
  * refuse new calls when within $5 of the cap (the "buffer")

Per-project soft caps (``project_soft_cap_usd`` in
``projects/<id>/config.toml``) are enforced by :func:`assert_project_budget`.
They are advisory — the monthly hard cap is always the final gate.

The ledger is append-only — every API charge (and every cache hit
recorded as $0) is one row in ``budget_ledger``. Spend is aggregated
per ``month_key`` (YYYY-MM) for fast monthly summing.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from autoscientist.state.db import month_key, now_iso


class BudgetExceeded(RuntimeError):
    """Raised when an estimated cost would push monthly spend past the cap minus buffer."""


@dataclass(frozen=True)
class BudgetConfig:
    monthly_cap_usd: float
    hard_floor_buffer_usd: float

    @property
    def refuse_threshold_usd(self) -> float:
        return self.monthly_cap_usd - self.hard_floor_buffer_usd

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BudgetConfig":
        cap = float(
            os.environ.get(
                "AUTOSCIENTIST_MONTHLY_CAP_USD", raw.get("monthly_cap_usd", 150.0)
            )
        )
        buffer = float(
            os.environ.get(
                "AUTOSCIENTIST_BUDGET_BUFFER_USD", raw.get("hard_floor_buffer_usd", 5.0)
            )
        )
        if cap <= 0 or buffer < 0:
            raise ValueError("monthly_cap_usd must be > 0 and buffer >= 0")
        return cls(monthly_cap_usd=cap, hard_floor_buffer_usd=buffer)


def monthly_spent(conn: sqlite3.Connection, month: str | None = None) -> float:
    """Return real (non-cache-hit) spend for the given YYYY-MM month."""
    month = month or month_key()
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) AS total FROM budget_ledger "
        "WHERE month_key = ? AND cache_hit = 0",
        (month,),
    ).fetchone()
    return float(row["total"])


def can_spend(
    conn: sqlite3.Connection,
    cfg: BudgetConfig,
    estimated_cost_usd: float,
    *,
    month: str | None = None,
) -> tuple[bool, float, float]:
    """Return ``(allowed, current_spend, projected_spend)``.

    ``allowed`` is False when ``projected_spend`` would exceed ``cap - buffer``.
    """
    spent = monthly_spent(conn, month)
    projected = spent + max(0.0, estimated_cost_usd)
    return (projected <= cfg.refuse_threshold_usd, spent, projected)


def assert_can_spend(
    conn: sqlite3.Connection,
    cfg: BudgetConfig,
    estimated_cost_usd: float,
    *,
    month: str | None = None,
) -> None:
    allowed, spent, projected = can_spend(conn, cfg, estimated_cost_usd, month=month)
    if not allowed:
        raise BudgetExceeded(
            f"refusing call: monthly_spent=${spent:.2f}, "
            f"estimated_charge=${estimated_cost_usd:.4f}, "
            f"projected=${projected:.2f}, cap=${cfg.monthly_cap_usd:.2f}, "
            f"buffer=${cfg.hard_floor_buffer_usd:.2f}"
        )


def record_charge(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    agent_name: str | None,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    cache_hit: bool,
) -> None:
    conn.execute(
        """INSERT INTO budget_ledger (
            run_id, agent_name, provider, model,
            prompt_tokens, completion_tokens, cost_usd, cache_hit,
            created_at, month_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, agent_name, provider, model,
            prompt_tokens, completion_tokens, float(cost_usd), int(cache_hit),
            now_iso(), month_key(),
        ),
    )


# ---------------------------------------------------------------------------
# Per-project budget enforcement
# ---------------------------------------------------------------------------

def project_spent(
    conn: sqlite3.Connection,
    project_id: str,
    *,
    month: str | None = None,
) -> float:
    """Return real (non-cache-hit) spend for ``project_id`` this month.

    Joins the ledger against ``runs`` to filter by project.
    """
    month = month or month_key()
    row = conn.execute(
        "SELECT COALESCE(SUM(bl.cost_usd), 0.0) AS total "
        "FROM budget_ledger bl "
        "JOIN runs r ON bl.run_id = r.run_id "
        "WHERE r.project_id = ? AND bl.month_key = ? AND bl.cache_hit = 0",
        (project_id, month),
    ).fetchone()
    return float(row["total"])


def assert_project_budget(
    conn: sqlite3.Connection,
    project_id: str,
    project_soft_cap_usd: float,
    estimated_cost_usd: float,
) -> None:
    """Raise :class:`BudgetExceeded` if projected project spend exceeds the soft cap.

    This is separate from the monthly hard cap and checked in addition to it.
    The soft cap comes from ``projects/<id>/config.toml [budget].project_soft_cap_usd``.
    """
    if project_soft_cap_usd <= 0:
        return  # no cap configured
    spent = project_spent(conn, project_id)
    projected = spent + max(0.0, estimated_cost_usd)
    if projected > project_soft_cap_usd:
        raise BudgetExceeded(
            f"project '{project_id}' soft cap exceeded: "
            f"spent=${spent:.2f}, estimated=${estimated_cost_usd:.4f}, "
            f"projected=${projected:.2f}, cap=${project_soft_cap_usd:.2f}"
        )
