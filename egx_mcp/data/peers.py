"""Peer comparison — relative valuation across the same EGX sector.

EGX is shallow; absolute multiples are misleading because the cost of
equity is sky-high. The honest read is always relative: COMI vs HDBK,
TMGH vs PHDC, not COMI vs the global banking index.

Returns the target ticker plus all sector peers with the same metrics
side-by-side, sorted by composite score so the best-of-sector floats
to the top.
"""
from __future__ import annotations

import logging
from typing import Any

from .fundamentals import get_fundamentals
from .scoring import score_stock
from .universe import EGX_UNIVERSE, resolve_ticker

log = logging.getLogger("egx-mcp.peers")


def compare(user_ticker: str, max_peers: int = 8) -> dict[str, Any]:
    """Return target + sector peers ranked by composite score."""
    canonical, _, _ = resolve_ticker(user_ticker)
    target_meta = EGX_UNIVERSE.get(canonical)
    if not target_meta:
        return {"error": f"{canonical} is not in the curated universe"}
    sector = target_meta["sector"]

    peers = [t for t, m in EGX_UNIVERSE.items()
             if m["sector"] == sector and m["sector"] != "Index"]

    rows = []
    for ticker in peers:
        try:
            f = get_fundamentals(ticker)
            sc = score_stock(ticker)
        except Exception as e:
            log.warning(f"peer fetch failed for {ticker}: {e}")
            continue
        if "error" in f or "error" in sc:
            continue
        rows.append({
            "ticker": ticker,
            "name": f.get("name"),
            "is_target": ticker == canonical,
            "price": f.get("price"),
            "pe": f.get("pe_ratio"),
            "pb": f.get("pb_ratio"),
            "roe_pct": f.get("roe_pct"),
            "margin_pct": f.get("profit_margin_pct"),
            "div_yield_pct": f.get("dividend_yield_pct"),
            "market_cap": f.get("market_cap"),
            "composite_score": sc.get("composite_score"),
        })

    rows.sort(key=lambda r: r.get("composite_score") or 0, reverse=True)
    rows = rows[:max(max_peers, 1)]

    target_row = next((r for r in rows if r["is_target"]), None)
    target_rank = next((i + 1 for i, r in enumerate(rows) if r["is_target"]), None)

    # Compute relative-value verdict for the target
    relative = None
    if target_row and target_row["composite_score"] is not None:
        scores = [r["composite_score"] for r in rows if r["composite_score"] is not None]
        if scores:
            avg = sum(scores) / len(scores)
            delta = target_row["composite_score"] - avg
            if delta >= 8:
                relative = "best_in_sector"
            elif delta >= 3:
                relative = "above_sector_average"
            elif delta <= -8:
                relative = "worst_in_sector"
            elif delta <= -3:
                relative = "below_sector_average"
            else:
                relative = "in_line_with_sector"

    return {
        "target": canonical,
        "sector": sector,
        "peer_count": len(rows),
        "target_rank_in_sector": target_rank,
        "target_relative_to_peers": relative,
        "peers": rows,
    }
