"""Sector for every EGX listing, not just the 30 curated names.

EGX_UNIVERSE (curated) always wins. With model_params.sector_source set to
"tradingview", names outside it get the model sector mapped from their
TradingView industry (egx_sectors.json), so they get sector medians and the
sector macro bias instead of "Unknown". Default "curated" keeps today's
behaviour.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from . import model_params
from .universe import EGX_UNIVERSE

_FILE = Path(__file__).parent / "egx_sectors.json"


@lru_cache(maxsize=1)
def _mapped() -> dict[str, str]:
    try:
        data = json.loads(_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {tk: v["sector"] for tk, v in data.get("tickers", {}).items() if v.get("sector")}


def all_sectors() -> dict[str, str]:
    """ticker -> sector over every classified listing, curated winning."""
    out = dict(_mapped())
    out.update({tk: m["sector"] for tk, m in EGX_UNIVERSE.items()
                if m.get("sector") and m["sector"] != "Index"})
    return out


def sector_of(ticker: str) -> str | None:
    cur = EGX_UNIVERSE.get(ticker, {}).get("sector")
    if cur:
        return cur
    if model_params.sector_source() == "tradingview":
        return _mapped().get(ticker)
    return None


def peers(sector: str) -> list[str]:
    return sorted(tk for tk, s in all_sectors().items() if s.lower() == sector.lower())
