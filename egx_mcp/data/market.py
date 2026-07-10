"""Market data adapter — yfinance-backed.

Yahoo Finance covers EGX with the .CA suffix. This module wraps yfinance
with EGX-aware ticker resolution and a small TTL cache to avoid hammering
the upstream API.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from ._certs import ensure_ca_bundle

ensure_ca_bundle()  # trust the OS cert store before yfinance opens any connection

import yfinance as yf

from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.market")

# Tiny in-memory TTL cache. For production, swap in diskcache or redis.
_CACHE: dict[str, tuple[float, Any]] = {}
_TTL_QUOTE = 60         # 1 min — quotes are 15-min delayed on Yahoo for EGX anyway
_TTL_HISTORY = 900      # 15 min
_TTL_INFO = 86400       # 24h — fundamentals don't move


def _cached(key: str, ttl: int, loader):
    now = time.time()
    if key in _CACHE:
        ts, val = _CACHE[key]
        if now - ts < ttl:
            return val
    val = loader()
    _CACHE[key] = (now, val)
    return val


def get_quote(user_ticker: str) -> dict[str, Any]:
    canonical, yahoo, name = resolve_ticker(user_ticker)

    def _load():
        t = yf.Ticker(yahoo)
        info = t.info or {}
        # fast_info has the live price fields
        fi = getattr(t, "fast_info", None)

        price = None
        prev_close = None
        day_high = None
        day_low = None
        volume = None
        try:
            price = fi["last_price"] if fi else info.get("currentPrice")
            prev_close = fi["previous_close"] if fi else info.get("previousClose")
            day_high = fi["day_high"] if fi else info.get("dayHigh")
            day_low = fi["day_low"] if fi else info.get("dayLow")
            volume = fi["last_volume"] if fi else info.get("volume")
        except Exception as e:
            log.warning(f"fast_info partial failure for {yahoo}: {e}")

        change = None
        change_pct = None
        if price is not None and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else None

        return {
            "ticker": canonical,
            "yahoo_symbol": yahoo,
            "name": info.get("longName") or info.get("shortName") or name,
            "price": price,
            "previous_close": prev_close,
            "change": round(change, 4) if change is not None else None,
            "change_pct": round(change_pct, 2) if change_pct is not None else None,
            "day_high": day_high,
            "day_low": day_low,
            "volume": volume,
            "avg_volume": info.get("averageVolume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "currency": info.get("currency", "EGP"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    return _cached(f"quote:{yahoo}", _TTL_QUOTE, _load)


def get_history(user_ticker: str, period: str = "6mo", interval: str = "1d") -> dict[str, Any]:
    canonical, yahoo, name = resolve_ticker(user_ticker)

    def _load():
        t = yf.Ticker(yahoo)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            return {
                "ticker": canonical,
                "yahoo_symbol": yahoo,
                "period": period,
                "interval": interval,
                "rows": [],
                "summary": None,
                "error": "No data returned. Ticker may be wrong or delisted.",
            }

        rows = []
        for idx, row in df.iterrows():
            rows.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0,
            })

        # Summary stats
        closes = df["Close"].dropna()
        start_price = float(closes.iloc[0])
        end_price = float(closes.iloc[-1])
        ret_pct = (end_price / start_price - 1) * 100

        # Max drawdown
        running_max = closes.cummax()
        drawdown = (closes / running_max - 1) * 100
        max_dd = float(drawdown.min())

        # Annualized volatility (daily returns)
        daily_returns = closes.pct_change().dropna()
        vol_pct = float(daily_returns.std() * (252 ** 0.5) * 100) if len(daily_returns) > 1 else None

        return {
            "ticker": canonical,
            "yahoo_symbol": yahoo,
            "name": name,
            "period": period,
            "interval": interval,
            "rows": rows,
            "summary": {
                "start_date": rows[0]["date"],
                "end_date": rows[-1]["date"],
                "start_price": round(start_price, 4),
                "end_price": round(end_price, 4),
                "return_pct": round(ret_pct, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "annualized_volatility_pct": round(vol_pct, 2) if vol_pct else None,
                "bar_count": len(rows),
            },
        }

    return _cached(f"history:{yahoo}:{period}:{interval}", _TTL_HISTORY, _load)


def get_index(name: str = "EGX30") -> dict[str, Any]:
    canonical, yahoo, full_name = resolve_ticker(name)
    quote = get_quote(name)

    # YTD return — Yahoo's historical depth for EGX indices is poor;
    # ^CASE30 typically returns only 1 bar via .history(). We attempt
    # the YTD calc but mark it null when data is unavailable.
    ytd_pct = None
    note = None
    try:
        from datetime import date
        start = date(date.today().year, 1, 1).strftime("%Y-%m-%d")
        df = yf.Ticker(yahoo).history(start=start, interval="1d", auto_adjust=False)
        if not df.empty and len(df) > 1:
            start_p = float(df["Close"].iloc[0])
            end_p = float(df["Close"].iloc[-1])
            ytd_pct = round((end_p / start_p - 1) * 100, 2)
        else:
            note = (
                "YTD return unavailable: Yahoo Finance has limited historical "
                "depth for EGX index symbols. Use individual stock returns or "
                "the EGX 30 ETF (EGS69491M015.CA) as a proxy if needed."
            )
    except Exception as e:
        log.warning(f"YTD calc failed for {yahoo}: {e}")
        note = f"YTD calc error: {e}"

    return {
        "index": canonical,
        "yahoo_symbol": yahoo,
        "name": full_name,
        "value": quote.get("price"),
        "change": quote.get("change"),
        "change_pct": quote.get("change_pct"),
        "day_high": quote.get("day_high"),
        "day_low": quote.get("day_low"),
        "ytd_return_pct": ytd_pct,
        "timestamp": quote.get("timestamp"),
        "note": note,
    }
