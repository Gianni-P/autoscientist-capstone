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
from datetime import UTC, datetime
from typing import Any

from autoscientist.state.db import month_key


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
    """Raise :class:`BudgetExceeded` if this charge would breach the monthly cap.

    This is a plain read-then-act check with no write, so under CONCURRENT
    writers two callers can both read the same stale total and both pass. The
    router no longer uses it on the hot path: :func:`reserve_charge` checks and
    writes the reservation atomically under one ``BEGIN IMMEDIATE`` lock, which
    is race-free. ``assert_can_spend`` remains for read-only "can I afford X?"
    queries and tests where no reservation is being made.
    """
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
    # Derive created_at and month_key from ONE instant so a charge logged at the
    # UTC month boundary can't land its timestamp and its month bucket in
    # different months.
    ts = datetime.now(UTC)
    conn.execute(
        """INSERT INTO budget_ledger (
            run_id, agent_name, provider, model,
            prompt_tokens, completion_tokens, cost_usd, cache_hit,
            created_at, month_key
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, agent_name, provider, model,
            prompt_tokens, completion_tokens, float(cost_usd), int(cache_hit),
            ts.isoformat(timespec="milliseconds"), month_key(ts),
        ),
    )


def reserve_charge(
    conn: sqlite3.Connection,
    *,
    run_id: str | None,
    agent_name: str | None,
    provider: str,
    model: str,
    estimated_cost_usd: float,
    cfg: BudgetConfig,
    month: str | None = None,
    project_id: str | None = None,
    project_soft_cap_usd: float = 0.0,
) -> int:
    """Atomically check the caps AND reserve ``estimated_cost_usd`` as a ledger row.

    This closes the read-then-write race in :func:`assert_can_spend`: the
    monthly (and optional per-project) cap is checked and the reservation is
    written inside a single ``BEGIN IMMEDIATE`` transaction, so two concurrent
    callers serialize on the write lock and the second re-reads the first's
    committed reservation before deciding. The reservation counts toward
    ``monthly_spent`` immediately; reconcile it to the real cost after the call
    with :func:`reconcile_charge`, or drop it with :func:`release_reservation`
    if the call fails.

    Returns the ledger row id. Raises :class:`BudgetExceeded` if the reservation
    would breach a cap (nothing is written in that case).
    """
    month = month or month_key()
    est = max(0.0, estimated_cost_usd)
    ts = datetime.now(UTC)
    # Close any open transaction so BEGIN IMMEDIATE is legal, then take the
    # write lock up front (before the read) so the check+insert is serialized.
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        spent = monthly_spent(conn, month)
        projected = spent + est
        if projected > cfg.refuse_threshold_usd:
            raise BudgetExceeded(
                f"refusing call: monthly_spent=${spent:.2f}, "
                f"estimated_charge=${est:.4f}, projected=${projected:.2f}, "
                f"cap=${cfg.monthly_cap_usd:.2f}, buffer=${cfg.hard_floor_buffer_usd:.2f}"
            )
        if project_id and project_soft_cap_usd > 0:
            pspent = project_spent(conn, project_id, month=month)
            if pspent + est > project_soft_cap_usd:
                raise BudgetExceeded(
                    f"project '{project_id}' soft cap exceeded: spent=${pspent:.2f}, "
                    f"estimated=${est:.4f}, projected=${pspent + est:.2f}, "
                    f"cap=${project_soft_cap_usd:.2f}"
                )
        cur = conn.execute(
            """INSERT INTO budget_ledger (
                run_id, agent_name, provider, model,
                prompt_tokens, completion_tokens, cost_usd, cache_hit,
                created_at, month_key
            ) VALUES (?, ?, ?, ?, 0, 0, ?, 0, ?, ?)""",
            (run_id, agent_name, provider, model, est,
             ts.isoformat(timespec="milliseconds"), month),
        )
        conn.commit()
        return int(cur.lastrowid)
    except BaseException:
        conn.rollback()
        raise


def reconcile_charge(
    conn: sqlite3.Connection,
    ledger_id: int,
    *,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Update a reservation row to the actual cost and token counts."""
    conn.execute(
        "UPDATE budget_ledger SET cost_usd = ?, prompt_tokens = ?, completion_tokens = ? "
        "WHERE ledger_id = ?",
        (float(cost_usd), int(prompt_tokens), int(completion_tokens), int(ledger_id)),
    )


def release_reservation(conn: sqlite3.Connection, ledger_id: int) -> None:
    """Drop a reservation row — e.g. when the LLM call failed before any spend."""
    conn.execute("DELETE FROM budget_ledger WHERE ledger_id = ?", (int(ledger_id),))


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
