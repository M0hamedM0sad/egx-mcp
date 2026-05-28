"""Fundamentals adapter — sanitized P/E, P/B, ROE, margins, leverage.

Yahoo's `trailingPE` for EGX names is unreliable (the README documents
CIRA returning 0.12). This module:

  1. Pulls raw fundamentals from yfinance (`info` + `fast_info`).
  2. Recomputes P/E from price / trailingEps when Yahoo's value looks bogus.
  3. Recomputes P/B from price / bookValue.
  4. **Optionally overrides** any field from a CSV at $EGX_FUNDAMENTALS_CSV
     so users can pipe in audited values from EGX disclosures, Bloomberg,
     or manual research without touching code.
  5. Returns a clean fundamental snapshot for scoring.

Audited override CSV format (case-insensitive headers):
    ticker,trailing_eps,book_value_per_share,roe_pct,profit_margin_pct,
    debt_to_equity,dividend_yield_pct,market_cap

Sector medians are derived from the universe at call time so the rubric
adapts as new names are added.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from pathlib import Path
from statistics import median
from typing import Any

import yfinance as yf

from .universe import EGX_UNIVERSE, resolve_ticker

log = logging.getLogger("egx-mcp.fundamentals")

# In-process TTL cache so repeated calls inside a single briefing run
# don't re-fetch (and re-fail-loudly for) the same yfinance object.
_FUND_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_FUND_CACHE_TTL_SECS = 600


# Sanity bounds — anything outside these is treated as a Yahoo data error.
_PE_MIN = 1.0
_PE_MAX = 200.0
_PB_MIN = 0.05
_PB_MAX = 50.0


def _safe_div(a, b):
    try:
        if a is None or b is None or b == 0:
            return None
        return a / b
    except (TypeError, ZeroDivisionError):
        return None


def _sanitize_pe(raw_pe, price, eps):
    """Return a P/E we trust, or None.

    If raw_pe is in the reasonable band, use it. Otherwise recompute
    from price / EPS if both are known and EPS > 0.
    """
    if raw_pe is not None and _PE_MIN <= raw_pe <= _PE_MAX:
        return round(raw_pe, 2)
    recomputed = _safe_div(price, eps)
    if recomputed is not None and _PE_MIN <= recomputed <= _PE_MAX:
        return round(recomputed, 2)
    return None


def _sanitize_pb(raw_pb, price, book_value):
    if raw_pb is not None and _PB_MIN <= raw_pb <= _PB_MAX:
        return round(raw_pb, 2)
    recomputed = _safe_div(price, book_value)
    if recomputed is not None and _PB_MIN <= recomputed <= _PB_MAX:
        return round(recomputed, 2)
    return None


# ---------------------------------------------------------------------------
# AUDITED OVERRIDE — load once per process, optionally
# ---------------------------------------------------------------------------

_OVERRIDE_CACHE: dict[str, dict] | None = None


def _load_overrides() -> dict[str, dict]:
    """Read CSV pointed to by $EGX_FUNDAMENTALS_CSV, return {ticker: row}."""
    global _OVERRIDE_CACHE
    if _OVERRIDE_CACHE is not None:
        return _OVERRIDE_CACHE
    # Env var wins; otherwise fall back to the committed CSV next to this
    # module so audited fundamentals load with zero configuration.
    path = os.environ.get("EGX_FUNDAMENTALS_CSV")
    if not path:
        # Default search order, so audited fundamentals load with zero config:
        #   1. the audited CSV at repo root — what scripts/export_fundamentals_csv.py
        #      writes from the Mubasher cache (the real, populated file);
        #   2. the legacy data-dir file (kept for back-compat).
        candidates = [
            Path(__file__).parent.parent.parent / "egx_fundamentals_audited.csv",
            Path(__file__).parent / "egx_fundamentals.csv",
        ]
        path = next((str(p) for p in candidates if p.exists()), None)
    if not path or not Path(path).exists():
        _OVERRIDE_CACHE = {}
        return _OVERRIDE_CACHE
    out: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row = {k.lower().strip(): v for k, v in row.items() if k}
                tk = (row.get("ticker") or "").strip().upper()
                if not tk:
                    continue
                parsed = {}
                for field in ("trailing_eps", "book_value_per_share", "roe_pct",
                              "profit_margin_pct", "debt_to_equity",
                              "dividend_yield_pct", "market_cap", "pe_ratio", "pb_ratio"):
                    val = row.get(field)
                    if val not in (None, ""):
                        try:
                            parsed[field] = float(val)
                        except ValueError:
                            pass
                if parsed:
                    out[tk] = parsed
        log.info(f"Loaded {len(out)} fundamental overrides from {path}")
    except Exception as e:
        log.warning(f"Failed to load fundamentals override CSV: {e}")
    _OVERRIDE_CACHE = out
    return _OVERRIDE_CACHE


def get_fundamentals(user_ticker: str) -> dict[str, Any]:
    """Return a sanitized fundamentals snapshot for a single ticker.

    Cached per-process for 10 minutes. yfinance can throw the same
    internal error on every retry inside a single briefing run (e.g.
    `'PriceHistory' object has no attribute '_dividends'` for IDHC),
    so we cache the exception payload too.
    """
    canonical, yahoo, name = resolve_ticker(user_ticker)

    cached = _FUND_CACHE.get(canonical)
    if cached and (time.time() - cached[0]) < _FUND_CACHE_TTL_SECS:
        return cached[1]

    # Pull live fundamentals from Yahoo. Yahoo blocks EGX `.CA` symbols in
    # many environments, so a failure here is expected — we fall back to the
    # audited CSV override (+ cached price) below rather than giving up.
    info: dict[str, Any] = {}
    price = None
    yahoo_ok = False
    yahoo_err: str | None = None
    try:
        t = yf.Ticker(yahoo)
        info = t.info or {}
        fi = getattr(t, "fast_info", None)
        price = (fi["last_price"] if fi else None) or info.get("currentPrice")
        yahoo_ok = True
    except Exception as e:
        yahoo_err = str(e)
        log.warning(f"fundamentals fetch failed for {yahoo}: {e}")

    # Offline price fallback so P/E can still be recomputed from audited EPS.
    if price is None:
        try:
            from . import price_cache
            q = price_cache.get_quote(canonical)
            if q and "error" not in q:
                price = q.get("price")
        except Exception:
            pass

    eps = info.get("trailingEps")
    book_value = info.get("bookValue")
    raw_pe = info.get("trailingPE")
    raw_pb = info.get("priceToBook")

    pe = _sanitize_pe(raw_pe, price, eps)
    pb = _sanitize_pb(raw_pb, price, book_value)

    roe = info.get("returnOnEquity")
    if roe is not None and abs(roe) < 5:
        roe_pct = round(roe * 100, 2)
    else:
        roe_pct = None

    profit_margin = info.get("profitMargins")
    pm_pct = round(profit_margin * 100, 2) if profit_margin is not None and abs(profit_margin) < 5 else None

    div_yield = info.get("dividendYield")
    if div_yield is not None:
        # Yahoo sometimes returns 0.05 (5%), sometimes 5.0 — normalize
        dy_pct = round(div_yield * 100, 2) if div_yield < 1 else round(div_yield, 2)
    else:
        dy_pct = None

    payload = {
        "ticker": canonical,
        "yahoo_symbol": yahoo,
        "name": info.get("longName") or info.get("shortName") or name,
        "sector": EGX_UNIVERSE.get(canonical, {}).get("sector"),
        "price": price,
        "trailing_eps": eps,
        "book_value_per_share": book_value,
        "pe_ratio": pe,
        "pb_ratio": pb,
        "raw_pe_from_yahoo": raw_pe,
        "pe_was_corrected": pe is not None and raw_pe is not None and abs((raw_pe - pe) / max(pe, 0.01)) > 0.1,
        "roe_pct": roe_pct,
        "profit_margin_pct": pm_pct,
        "debt_to_equity": info.get("debtToEquity"),
        "dividend_yield_pct": dy_pct,
        "market_cap": info.get("marketCap"),
        "forward_pe": info.get("forwardPE") if info.get("forwardPE") and info["forwardPE"] > 0 else None,
        "earnings_growth_yoy": info.get("earningsGrowth"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "fundamentals_source": "yahoo" if yahoo_ok else None,
    }

    # Apply audited override if present — stamp every overridden field.
    # This runs even when Yahoo failed, so an audited CSV is a complete
    # offline fundamentals source for EGX names Yahoo won't serve.
    overrides = _load_overrides().get(canonical)
    if overrides:
        applied = []
        for k, v in overrides.items():
            if k in payload:
                payload[k] = v
                applied.append(k)
        # Re-derive P/E and P/B if EPS or book value were overridden
        if price and overrides.get("trailing_eps"):
            payload["pe_ratio"] = _sanitize_pe(None, price, overrides["trailing_eps"])
        if price and overrides.get("book_value_per_share"):
            payload["pb_ratio"] = _sanitize_pb(None, price, overrides["book_value_per_share"])
        payload["fundamentals_source"] = "audited_override"
        payload["fields_overridden"] = applied

    # If Yahoo failed and there was no audited override to fall back on,
    # there's genuinely no fundamental data — surface the error.
    if not yahoo_ok and not overrides:
        payload_err = {
            "ticker": canonical,
            "name": name,
            "error": yahoo_err or "no fundamentals available (Yahoo blocked, no audited override)",
        }
        _FUND_CACHE[canonical] = (time.time(), payload_err)
        return payload_err

    _FUND_CACHE[canonical] = (time.time(), payload)
    return payload


def sector_medians(sector: str) -> dict[str, Any]:
    """Compute sector-level median P/E, P/B, ROE for relative valuation.

    Walks the curated universe — slow on cold cache, fast on warm cache
    because get_fundamentals shares yfinance's underlying response.
    """
    peers = [t for t, m in EGX_UNIVERSE.items()
             if m["sector"].lower() == sector.lower() and m["sector"] != "Index"]

    pes, pbs, roes, margins = [], [], [], []
    for ticker in peers:
        try:
            f = get_fundamentals(ticker)
        except Exception:
            continue
        if f.get("pe_ratio") is not None:
            pes.append(f["pe_ratio"])
        if f.get("pb_ratio") is not None:
            pbs.append(f["pb_ratio"])
        if f.get("roe_pct") is not None:
            roes.append(f["roe_pct"])
        if f.get("profit_margin_pct") is not None:
            margins.append(f["profit_margin_pct"])

    return {
        "sector": sector,
        "peer_count": len(peers),
        "median_pe": round(median(pes), 2) if pes else None,
        "median_pb": round(median(pbs), 2) if pbs else None,
        "median_roe_pct": round(median(roes), 2) if roes else None,
        "median_margin_pct": round(median(margins), 2) if margins else None,
        "n_with_pe": len(pes),
        "n_with_pb": len(pbs),
    }
