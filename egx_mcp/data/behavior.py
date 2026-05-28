"""Stock behavior decomposition — what actually moves each EGX name?

Most of the MCP answers "should I buy this?". This module answers a
different question: "what *drives* this stock?" For one ticker or the
whole universe, it combines four lenses into a single profile:

  1. Macro factor betas   — market / EGP / oil / gold / EM (via `factors`)
  2. Systematic vs idiosyncratic share — R² of the factor model. A high
     R² means the name is a macro puppet; a low R² means it dances to
     its own (stock-specific / news-driven) tune.
  3. Risk character        — annualized vol, max drawdown, trailing return
  4. Fundamental anchor     — P/E, P/B, ROE, leverage (via `fundamentals`)

The universe scan then rolls these up by sector so you can see, e.g.,
"banks are low-beta, EGP-strength names with tight idiosyncratic spread"
vs "petrochems are oil-levered and mostly macro-driven".

Coverage caveat: EGX has ~240 listings but Yahoo only carries reliable
daily history for the validated extended set (~68 names, see
`egx_listing`). Names without usable history are reported under
`skipped`, not silently dropped.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from . import factors, fundamentals, market, price_cache
from .universe import EGX_UNIVERSE, resolve_ticker

log = logging.getLogger("egx-mcp.behavior")


# Sector taxonomy for the validated extended universe. EGX_UNIVERSE only
# covers the ~30 curated names (and under nicknames like SODIC/EIPI), so
# this map fills the rest using the groupings documented in egx_listing.
_SECTOR_MAP: dict[str, str] = {
    # Banks
    "COMI": "Banks", "HDBK": "Banks", "CIEB": "Banks", "ADIB": "Banks",
    "FAIT": "Banks", "EGBE": "Banks", "SAUD": "Banks", "EXPA": "Banks",
    # Non-bank financials
    "HRHO": "Financial Services", "EFIH": "Financial Services",
    "CCAP": "Financial Services", "BTFH": "Financial Services",
    "OFH": "Financial Services",
    # Real estate
    "TMGH": "Real Estate", "PHDC": "Real Estate", "ORHD": "Real Estate",
    "EMFD": "Real Estate", "OCDI": "Real Estate", "HELI": "Real Estate",
    "AMER": "Real Estate", "OBRI": "Real Estate", "UEGC": "Real Estate",
    "RREI": "Real Estate", "MASR": "Real Estate", "ZMID": "Real Estate",
    # Construction / building materials
    "ORAS": "Construction", "ARCC": "Building Materials",
    "MISR": "Building Materials", "SCEM": "Building Materials",
    "MICH": "Building Materials", "CERA": "Building Materials",
    "LCSW": "Building Materials",
    # Basic resources
    "IRON": "Basic Resources",
    # Industrial goods
    "SWDY": "Industrial Goods", "ELEC": "Industrial Goods",
    "DSCW": "Industrial Goods", "ROTO": "Industrial Goods",
    "GBCO": "Automotive",
    # Chemicals / petrochems
    "ABUK": "Chemicals", "MFPC": "Chemicals", "AMOC": "Chemicals",
    "SKPC": "Chemicals", "EFIC": "Chemicals", "EGCH": "Chemicals",
    # Food & beverage / consumer
    "EFID": "Food & Beverage", "JUFO": "Food & Beverage",
    "DOMT": "Food & Beverage", "POUL": "Food & Beverage",
    "OLFI": "Food & Beverage", "MOIL": "Food & Beverage",
    "SUGR": "Food & Beverage",
    "ORWE": "Personal & Household", "EAST": "Personal & Household",
    "ASCM": "Personal & Household",
    # Telecom / tech
    "ETEL": "Telecom", "FWRY": "Technology", "RAYA": "Technology",
    "MTIE": "Retail",
    # Healthcare
    "CLHO": "Healthcare", "PHAR": "Healthcare", "BIOC": "Healthcare",
    "ACGC": "Healthcare",
    # Tourism / transport
    "EGTS": "Travel & Leisure", "ALCN": "Transport", "CSAG": "Transport",
    "ETRS": "Transport",
    # Education
    "CIRA": "Education",
    # Diversified / misc
    "BINV": "Diversified",
}

_FACTOR_LABELS = {
    "egp": "EGP/USD (FX)",
    "brent": "Brent crude (oil)",
    "gold": "Gold (safe-haven)",
    "em": "EM equities (regional risk)",
}

# yfinance only accepts a fixed set of period strings.
_PERIOD_LADDER = [(45, "3mo"), (140, "6mo"), (300, "1y"), (600, "2y")]


def _sector_for(ticker: str) -> str:
    tk = ticker.upper()
    if tk in _SECTOR_MAP:
        return _SECTOR_MAP[tk]
    meta = EGX_UNIVERSE.get(tk)
    if meta and meta.get("sector") and meta["sector"] != "Index":
        return meta["sector"]
    return "Unknown"


def _period_for(lookback_days: int) -> str:
    for max_days, period in _PERIOD_LADDER:
        if lookback_days <= max_days:
            return period
    return "max"


def _risk_metrics(ticker: str, lookback_days: int) -> dict[str, Any]:
    """Trailing return / volatility / drawdown from price history."""
    try:
        hist = market.get_history(ticker, period=_period_for(lookback_days))
    except Exception as e:
        return {"error": f"history fetch failed: {e}"}
    summary = hist.get("summary")
    if not summary:
        return {"error": hist.get("error", "no history")}
    return {
        "trailing_return_pct": summary.get("return_pct"),
        "annualized_volatility_pct": summary.get("annualized_volatility_pct"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "bar_count": summary.get("bar_count"),
    }


def _vol_bucket(vol_pct: float | None) -> str | None:
    if vol_pct is None:
        return None
    if vol_pct < 25:
        return "low"
    if vol_pct < 45:
        return "moderate"
    if vol_pct < 70:
        return "high"
    return "very high"


def _dominant_macro_factor(betas: dict[str, float]) -> tuple[str | None, float]:
    """Largest non-market factor by absolute beta."""
    best, best_abs = None, 0.0
    for f in ("egp", "brent", "gold", "em"):
        b = betas.get(f)
        if b is None:
            continue
        if abs(b) > best_abs:
            best, best_abs = f, abs(b)
    return best, best_abs


def _behavior_notes(
    betas: dict[str, float],
    r2: float | None,
    risk: dict[str, Any],
    fund: dict[str, Any],
) -> list[str]:
    notes: list[str] = []

    mkt = betas.get("egx30")
    if mkt is not None:
        if mkt >= 1.2:
            notes.append(f"High-beta — amplifies the market (EGX30 β={mkt:.2f}).")
        elif mkt <= 0.5:
            notes.append(f"Defensive — muted vs the market (EGX30 β={mkt:.2f}).")
        else:
            notes.append(f"Tracks the market roughly 1:1 (EGX30 β={mkt:.2f}).")

    dom, dom_abs = _dominant_macro_factor(betas)
    if dom and dom_abs >= 0.2:
        direction = "rises with" if betas[dom] > 0 else "falls with"
        notes.append(f"Most macro-sensitive to {_FACTOR_LABELS[dom]} — {direction} it (β={betas[dom]:.2f}).")
        if dom == "egp":
            if betas["egp"] > 0:
                notes.append("EGP-weakening winner — exporter / hard-currency revenue profile.")
            else:
                notes.append("EGP-strength winner — importer / EGP cost base.")

    if r2 is not None:
        idio = (1 - r2) * 100
        if r2 >= 0.5:
            notes.append(f"Mostly macro-driven — factors explain {r2*100:.0f}% of moves; stock-specific share only {idio:.0f}%.")
        elif r2 <= 0.2:
            notes.append(f"Largely idiosyncratic — {idio:.0f}% of moves are stock-specific (news / disclosures / flows), not macro.")
        else:
            notes.append(f"Mixed driver — macro explains {r2*100:.0f}%, the rest ({idio:.0f}%) is stock-specific.")

    vb = _vol_bucket(risk.get("annualized_volatility_pct"))
    if vb:
        notes.append(f"{vb.capitalize()} volatility ({risk['annualized_volatility_pct']:.0f}% annualized).")

    pe = fund.get("pe_ratio")
    roe = fund.get("roe_pct")
    if pe is not None and roe is not None:
        notes.append(f"Fundamental anchor: P/E {pe:.1f}, ROE {roe:.0f}%.")
    elif pe is not None:
        notes.append(f"Fundamental anchor: P/E {pe:.1f}.")

    return notes


def _profile_from_cache(canonical: str, name: str, drivers: dict[str, Any]) -> dict[str, Any]:
    """Build a behavior profile from precomputed cache drivers (offline path)."""
    betas = drivers.get("factor_betas", {})
    r2 = drivers.get("r_squared")
    risk = {
        "trailing_return_pct": drivers.get("trailing_return_pct"),
        "annualized_volatility_pct": drivers.get("annualized_volatility_pct"),
        "max_drawdown_pct": drivers.get("max_drawdown_pct"),
        "momentum_20d_pct": drivers.get("momentum_20d_pct"),
        "momentum_60d_pct": drivers.get("momentum_60d_pct"),
        "bar_count": drivers.get("n_obs"),
    }

    fund_raw = fundamentals.get_fundamentals(canonical)
    fund = {
        "pe_ratio": fund_raw.get("pe_ratio"),
        "pb_ratio": fund_raw.get("pb_ratio"),
        "roe_pct": fund_raw.get("roe_pct"),
        "debt_to_equity": fund_raw.get("debt_to_equity"),
        "dividend_yield_pct": fund_raw.get("dividend_yield_pct"),
    } if "error" not in fund_raw else {"error": fund_raw["error"]}

    dom, _ = _dominant_macro_factor(betas)

    return {
        "ticker": canonical,
        "name": fund_raw.get("name") or name,
        "sector": _sector_for(canonical),
        "lookback_days": drivers.get("n_obs"),
        "source": "price_cache",
        "drivers": {
            "market_beta": drivers.get("market_beta"),
            "factor_betas": betas,
            "dominant_macro_factor": dom,
            "systematic_r2": r2,
            "idiosyncratic_pct": drivers.get("idiosyncratic_pct"),
            "alpha_daily_pct": drivers.get("alpha_daily_pct"),
        },
        "risk": risk,
        "fundamentals": fund,
        "interpretation": _behavior_notes(betas, r2, risk, fund),
    }


def stock_behavior(user_ticker: str, lookback_days: int = 120) -> dict[str, Any]:
    """Full behavior profile for a single EGX name.

    Prefers the offline price_cache (precomputed drivers from investing.com)
    when present; falls back to the live factor regression otherwise.
    """
    canonical, yahoo, name = resolve_ticker(user_ticker)

    cached = price_cache.get_drivers(canonical)
    if cached and "error" not in cached:
        return _profile_from_cache(canonical, name, cached)

    fx = factors.ticker_factor_exposure(user_ticker, lookback_days=lookback_days)
    if "error" in fx:
        return {"ticker": canonical, "name": name, "sector": _sector_for(canonical),
                "error": fx["error"]}

    betas = fx.get("factor_betas", {})
    r2 = fx.get("r_squared")
    risk = _risk_metrics(user_ticker, lookback_days)

    fund_raw = fundamentals.get_fundamentals(user_ticker)
    fund = {
        "pe_ratio": fund_raw.get("pe_ratio"),
        "pb_ratio": fund_raw.get("pb_ratio"),
        "roe_pct": fund_raw.get("roe_pct"),
        "debt_to_equity": fund_raw.get("debt_to_equity"),
        "dividend_yield_pct": fund_raw.get("dividend_yield_pct"),
    } if "error" not in fund_raw else {"error": fund_raw["error"]}

    dom, _ = _dominant_macro_factor(betas)

    return {
        "ticker": canonical,
        "name": fund_raw.get("name") or name,
        "sector": _sector_for(canonical),
        "lookback_days": fx.get("lookback_days"),
        "drivers": {
            "market_beta": betas.get("egx30"),
            "factor_betas": betas,
            "dominant_macro_factor": dom,
            "systematic_r2": r2,
            "idiosyncratic_pct": round((1 - r2) * 100, 1) if r2 is not None else None,
            "alpha_daily_pct": fx.get("alpha_daily_pct"),
        },
        "risk": risk,
        "fundamentals": fund,
        "interpretation": _behavior_notes(betas, r2, risk, fund),
    }


def _universe_symbols(universe: str) -> list[str]:
    if universe == "curated":
        return [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]
    # default: validated extended set
    from .egx_listing import get_full_universe
    syms = get_full_universe()
    return syms or [t for t, m in EGX_UNIVERSE.items() if m["sector"] != "Index"]


def _avg(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def scan_universe_behavior(
    universe: str = "extended",
    lookback_days: int = 120,
    sector: str | None = None,
) -> dict[str, Any]:
    """Profile every reachable EGX name and roll up by sector.

    Args:
        universe: 'extended' (~68 validated) or 'curated' (~30 named).
        lookback_days: Factor-regression and risk window.
        sector: Optional sector filter (case-insensitive substring).
    """
    symbols = _universe_symbols(universe)

    profiles: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for sym in symbols:
        if sector and sector.lower() not in _sector_for(sym).lower():
            continue
        prof = stock_behavior(sym, lookback_days=lookback_days)
        if "error" in prof:
            skipped.append({"ticker": prof["ticker"], "reason": prof["error"]})
            continue
        profiles.append(prof)

    # Sector roll-up
    by_sector: dict[str, dict[str, Any]] = {}
    sector_groups: dict[str, list[dict[str, Any]]] = {}
    for p in profiles:
        sector_groups.setdefault(p["sector"], []).append(p)

    for sec, members in sector_groups.items():
        betas_sum: dict[str, list[float]] = {}
        for m in members:
            for f, b in m["drivers"]["factor_betas"].items():
                betas_sum.setdefault(f, []).append(b)
        avg_betas = {f: _avg(v) for f, v in betas_sum.items()}
        dom, _ = _dominant_macro_factor({k: (v or 0) for k, v in avg_betas.items()})
        by_sector[sec] = {
            "n": len(members),
            "avg_market_beta": avg_betas.get("egx30"),
            "dominant_macro_factor": dom,
            "avg_factor_betas": avg_betas,
            "avg_volatility_pct": _avg([m["risk"].get("annualized_volatility_pct") for m in members]),
            "avg_idiosyncratic_pct": _avg([m["drivers"].get("idiosyncratic_pct") for m in members]),
            "tickers": [m["ticker"] for m in members],
        }

    profiles.sort(key=lambda p: (p["sector"], p["ticker"]))

    return {
        "as_of": datetime.utcnow().isoformat() + "Z",
        "universe": universe,
        "lookback_days": lookback_days,
        "sector_filter": sector,
        "n_scanned": len(symbols),
        "n_with_data": len(profiles),
        "n_skipped": len(skipped),
        "by_sector": dict(sorted(by_sector.items())),
        "stocks": profiles,
        "skipped": skipped,
        "coverage_note": (
            "EGX has ~240 listings; only names with reliable Yahoo Finance "
            "daily history are profiled. Illiquid / unlisted-on-Yahoo names "
            "appear under 'skipped' or are absent from the validated universe."
        ),
    }
