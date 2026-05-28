"""Catalyst calendar — earnings, dividends, disclosure clusters.

A decision shouldn't be made without knowing what's about to move the
name. We pull three forward-looking signals:

  1. Next earnings date (yfinance Ticker.calendar)
  2. Ex-dividend date (yfinance info)
  3. Disclosure clustering — whether the name has been unusually active
     on the EGX disclosure portal in the last 14 days (a leading
     indicator of capital actions or material events)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

from . import disclosures
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.calendar")


def _safe_date(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, str):
        return value[:10]
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value).strftime("%Y-%m-%d")
        except Exception:
            return None
    return None


def _days_until(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return (d - datetime.utcnow()).days
    except ValueError:
        return None


def get_calendar(user_ticker: str, days_lookback: int = 14) -> dict[str, Any]:
    """Return next earnings, ex-dividend, and disclosure activity profile."""
    canonical, yahoo, name = resolve_ticker(user_ticker)

    earnings_date = None
    ex_div_date = None
    div_amount = None

    try:
        t = yf.Ticker(yahoo)
        cal = getattr(t, "calendar", None)
        if cal is not None:
            # Newer yfinance returns a dict; older returns a DataFrame
            if isinstance(cal, dict):
                earnings_date = _safe_date(
                    (cal.get("Earnings Date") or [None])[0] if isinstance(cal.get("Earnings Date"), list)
                    else cal.get("Earnings Date")
                )
            else:
                try:
                    earnings_date = _safe_date(cal.loc["Earnings Date"].iloc[0])
                except Exception:
                    pass

        info = t.info or {}
        ex_div_date = _safe_date(info.get("exDividendDate"))
        div_amount = info.get("lastDividendValue") or info.get("dividendRate")
    except Exception as e:
        log.warning(f"calendar fetch failed for {yahoo}: {e}")

    # Disclosure activity in the lookback window
    disc_count = 0
    recent_disclosures = []
    try:
        d = disclosures.fetch(ticker=canonical, days=days_lookback)
        disc_count = d.get("count", 0)
        recent_disclosures = (d.get("disclosures") or [])[:5]
    except Exception as e:
        log.warning(f"disclosure clustering failed for {canonical}: {e}")

    # Catalyst flags — these become blocking signals in the decision layer
    flags = []
    days_to_earnings = _days_until(earnings_date)
    days_to_exdiv = _days_until(ex_div_date)

    if days_to_earnings is not None and 0 <= days_to_earnings <= 7:
        flags.append({
            "type": "earnings_imminent",
            "severity": "high",
            "days": days_to_earnings,
            "message": f"Earnings in {days_to_earnings} days — defer initiation, hold existing"
        })
    elif days_to_earnings is not None and 0 <= days_to_earnings <= 14:
        flags.append({
            "type": "earnings_near",
            "severity": "medium",
            "days": days_to_earnings,
            "message": f"Earnings in {days_to_earnings} days — size down or wait"
        })

    if days_to_exdiv is not None and 0 <= days_to_exdiv <= 5:
        flags.append({
            "type": "exdiv_imminent",
            "severity": "low",
            "days": days_to_exdiv,
            "message": f"Goes ex-dividend in {days_to_exdiv} days — expect price gap"
        })

    if disc_count >= 5:
        flags.append({
            "type": "disclosure_cluster",
            "severity": "medium",
            "count": disc_count,
            "message": f"{disc_count} disclosures in {days_lookback}d — unusual activity, read before acting"
        })

    return {
        "ticker": canonical,
        "next_earnings_date": earnings_date,
        "days_to_earnings": days_to_earnings,
        "ex_dividend_date": ex_div_date,
        "days_to_exdividend": days_to_exdiv,
        "last_dividend_egp": div_amount,
        "disclosure_count_window": disc_count,
        "disclosure_window_days": days_lookback,
        "recent_disclosures": recent_disclosures,
        "catalyst_flags": flags,
        "blocking": any(f["severity"] == "high" for f in flags),
    }
