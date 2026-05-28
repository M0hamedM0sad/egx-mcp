"""Portfolio adapter — reads a local CSV, computes live P&L.

CSV format (columns are case-insensitive):
    ticker, shares, cost_basis
Optional:
    purchase_date, account, notes

Example:
    ticker,shares,cost_basis,purchase_date,notes
    CIRA,500,12.50,2024-09-15,
    COMI,100,72.30,2024-11-02,Bought after dividend
    SWDY,200,4.85,2025-01-20,
"""
from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Any

from . import market

log = logging.getLogger("egx-mcp.portfolio")


def _resolve_csv_path(csv_path: str | None) -> Path:
    if csv_path:
        return Path(csv_path).expanduser().resolve()
    env_path = os.environ.get("EGX_PORTFOLIO_CSV")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path.home() / "egx_portfolio.csv"


def summary(csv_path: str | None = None) -> dict[str, Any]:
    path = _resolve_csv_path(csv_path)
    if not path.exists():
        return {
            "error": f"Portfolio CSV not found at {path}.",
            "hint": (
                "Create a CSV with columns ticker,shares,cost_basis at "
                f"{path}, set $EGX_PORTFOLIO_CSV, or pass csv_path explicitly. "
                "Example row: CIRA,500,12.50"
            ),
        }

    positions_raw = []
    try:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize keys
                row = {k.lower().strip(): v for k, v in row.items() if k}
                ticker = (row.get("ticker") or "").strip()
                if not ticker:
                    continue
                try:
                    shares = float(row.get("shares") or 0)
                    cost = float(row.get("cost_basis") or 0)
                except ValueError:
                    log.warning(f"Skipping malformed row: {row}")
                    continue
                positions_raw.append({
                    "ticker": ticker,
                    "shares": shares,
                    "cost_basis": cost,
                    "purchase_date": (row.get("purchase_date") or None),
                    "account": (row.get("account") or None),
                    "notes": (row.get("notes") or None),
                })
    except Exception as e:
        return {"error": f"Failed to parse CSV at {path}: {e}"}

    if not positions_raw:
        return {"error": f"CSV at {path} is empty or has no valid rows."}

    # Pull live prices
    positions = []
    total_cost = 0.0
    total_value = 0.0
    failures = []

    for p in positions_raw:
        try:
            quote = market.get_quote(p["ticker"])
            price = quote.get("price")
            name = quote.get("name", p["ticker"])
        except Exception as e:
            failures.append({"ticker": p["ticker"], "error": str(e)})
            price = None
            name = p["ticker"]

        cost_total = p["shares"] * p["cost_basis"]
        market_value = p["shares"] * price if price is not None else None
        unrealized = (market_value - cost_total) if market_value is not None else None
        unrealized_pct = (unrealized / cost_total * 100) if (unrealized is not None and cost_total) else None

        total_cost += cost_total
        if market_value is not None:
            total_value += market_value

        positions.append({
            "ticker": p["ticker"],
            "name": name,
            "shares": p["shares"],
            "cost_basis": round(p["cost_basis"], 4),
            "current_price": round(price, 4) if price is not None else None,
            "cost_total": round(cost_total, 2),
            "market_value": round(market_value, 2) if market_value is not None else None,
            "unrealized_pnl": round(unrealized, 2) if unrealized is not None else None,
            "unrealized_pnl_pct": round(unrealized_pct, 2) if unrealized_pct is not None else None,
            "purchase_date": p["purchase_date"],
            "account": p["account"],
            "notes": p["notes"],
        })

    # Compute weights
    for pos in positions:
        if pos["market_value"] and total_value:
            pos["weight_pct"] = round(pos["market_value"] / total_value * 100, 2)
        else:
            pos["weight_pct"] = None

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

    # Sort by market value desc
    positions.sort(key=lambda p: p["market_value"] or 0, reverse=True)

    return {
        "csv_path": str(path),
        "position_count": len(positions),
        "total_cost_egp": round(total_cost, 2),
        "total_value_egp": round(total_value, 2),
        "total_pnl_egp": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "positions": positions,
        "failures": failures or None,
    }
