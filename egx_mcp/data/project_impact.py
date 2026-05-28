"""Project / catalyst impact — how news of future projects may move a stock.

Two halves, glued together:

  1. Event-study engine (offline, from the price cache). Using each name's
     factor model (alpha + betas on EGX30 / EGP / Brent / gold / EM), we
     strip the macro component out of every daily return to get the
     *abnormal return* — the stock-specific move that news, disclosures and
     flows drive. The distribution of those abnormal returns is the stock's
     empirical "reaction profile": how big a stock-specific move actually
     looks, calibrated on its own history.

  2. Catalyst scanner (online, from the news scrapers). Recent headlines are
     filtered to project / catalyst items (new developments, contracts, land
     deals, capital raises, …) and scored for tone. Each material item's
     expected price impact is direction (tone) × magnitude (the stock's
     historical typical catalyst move), with a low/high band.

The output is explicitly a *statistical reaction estimate*, not a forecast:
"a material positive catalyst on this name has historically moved it ~X%".
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

import numpy as np

from . import price_cache, sentiment
from .price_cache import _closes_by_date, _ffill_on, _returns, _FACTOR_KEYS
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.project_impact")

# Headline terms that mark a forward-looking project / catalyst. Tone comes
# from the sentiment lexicon; this set is only about *relevance*.
_CATALYST_EN = {
    "project", "projects", "launch", "launches", "develop", "development",
    "contract", "contracts", "awarded", "award", "win", "wins", "won",
    "sign", "signs", "signed", "mou", "agreement", "deal", "investment",
    "invest", "invests", "expansion", "expand", "expands", "land", "plot",
    "acquire", "acquires", "acquisition", "stake", "joint venture", "jv",
    "phase", "units", "backlog", "sukuk", "bond", "bonds", "capital increase",
    "factory", "plant", "mega", "city", "compound", "tender", "build",
    "construct", "construction", "billion", "bn", "egp",
}
_CATALYST_AR = {
    "مشروع", "مشروعات", "مشروعا", "إطلاق", "تطوير", "عقد", "عقود", "توقيع",
    "وقعت", "اتفاقية", "صفقة", "استثمار", "استثمارات", "توسع", "توسعات",
    "أرض", "أراضي", "قطعة", "استحواذ", "حصة", "مرحلة", "مدينة", "كمبوند",
    "مصنع", "محطة", "مناقصة", "إنشاء", "بناء", "مليار", "سداد", "سندات",
    "صكوك", "زيادة رأس المال",
}

_AR_RE = re.compile(r"[؀-ۿ]")


def _is_catalyst(title: str) -> bool:
    if not title:
        return False
    if _AR_RE.search(title):
        return any(term in title for term in _CATALYST_AR)
    toks = set(re.findall(r"[a-z]+", title.lower()))
    if toks & _CATALYST_EN:
        return True
    return bool(re.search(r"\d", title) and ("egp" in title.lower() or "bn" in toks))


# ---------------------------------------------------------------------------
# Event-study engine — abnormal returns from the cached factor model
# ---------------------------------------------------------------------------

def _abnormal_returns(ticker: str) -> tuple[list[str], np.ndarray] | None:
    """Daily abnormal return % = actual − (alpha + Σ beta·factor), from cache."""
    cache = price_cache._load()
    if not cache:
        return None
    rows = cache.get("prices", {}).get(ticker)
    drivers = cache.get("drivers", {}).get(ticker)
    if not rows or not drivers or "factor_betas" not in drivers:
        return None

    dates = [r["date"] for r in rows]
    y = _returns([r["close"] for r in rows])  # fraction, aligns to dates[1:]

    betas = drivers["factor_betas"]
    alpha = (drivers.get("alpha_daily_pct") or 0.0) / 100.0
    factor_prices = cache.get("factors", {})

    expected = np.full(len(y), alpha, dtype=float)
    for key in _FACTOR_KEYS:
        b = betas.get(key)
        frows = factor_prices.get(key)
        if not b or not frows:
            continue
        series = _closes_by_date(frows)
        fr = _returns(_ffill_on(dates, series))
        fr = np.nan_to_num(fr, nan=0.0)
        expected = expected + b * fr

    ar = (y - expected) * 100.0  # percent
    ar = np.nan_to_num(ar, nan=0.0)
    return dates[1:], ar


def _reaction_profile(ticker: str, drivers: dict[str, Any]) -> dict[str, Any]:
    out = _abnormal_returns(ticker)
    if out is None or len(out[1]) < 20:
        return {"error": "insufficient cached history for an event study"}
    ar_dates, ar = out
    abs_ar = np.abs(ar)
    p90 = float(np.percentile(abs_ar, 90))
    p75 = float(np.percentile(abs_ar, 75))
    p95 = float(np.percentile(abs_ar, 95))
    std = float(np.std(ar))

    # Biggest stock-specific moves (de-facto historical event days)
    order = np.argsort(abs_ar)[::-1][:5]
    biggest = [
        {"date": ar_dates[i], "abnormal_pct": round(float(ar[i]), 2)}
        for i in order
    ]
    n_events = int(np.sum(abs_ar > 2 * std))

    return {
        "n_obs": int(len(ar)),
        "mean_abs_abnormal_pct": round(float(np.mean(abs_ar)), 2),
        "std_abnormal_pct": round(std, 2),
        "typical_catalyst_move_pct": round(p90, 2),
        "catalyst_band_pct": [round(p75, 2), round(p95, 2)],
        "n_event_days_2sigma": n_events,
        "idiosyncratic_pct": drivers.get("idiosyncratic_pct"),
        "biggest_abnormal_moves": biggest,
    }


# ---------------------------------------------------------------------------
# Catalyst scanner — recent project news × the reaction profile
# ---------------------------------------------------------------------------

def _scan_catalysts(canonical: str, profile: dict[str, Any], limit: int) -> dict[str, Any]:
    typical = profile.get("typical_catalyst_move_pct")
    band = profile.get("catalyst_band_pct") or [None, None]

    senti = sentiment.analyze_sentiment(canonical, lang="both", limit=limit)
    catalysts: list[dict[str, Any]] = []
    for h in senti.get("headlines", []):
        title = h.get("title") or ""
        if not _is_catalyst(title):
            continue
        score = h.get("score") or 0.0
        direction = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
        sign = 1 if score > 0.05 else -1 if score < -0.05 else 0
        est = round(sign * typical, 2) if (typical is not None and sign) else 0.0
        est_range = (
            [round(sign * band[0], 2), round(sign * band[1], 2)]
            if (sign and band[0] is not None) else None
        )
        catalysts.append({
            "date": h.get("date"),
            "source": h.get("source"),
            "title": title,
            "url": h.get("url"),
            "tone": direction,
            "sentiment_score": score,
            "est_impact_pct": est,
            "est_impact_range_pct": est_range,
        })

    dirs = [1 if c["sentiment_score"] > 0.05 else -1 if c["sentiment_score"] < -0.05 else 0
            for c in catalysts]
    net_dir = (sum(dirs) / len(dirs)) if dirs else 0.0
    net_expected = round(net_dir * typical, 2) if (typical is not None and dirs) else 0.0
    net_tone = "bullish" if net_dir > 0.15 else "bearish" if net_dir < -0.15 else "mixed/neutral"

    return {
        "headlines_scanned": senti.get("headline_count", 0),
        "n_catalysts": len(catalysts),
        "net_tone": net_tone,
        "net_expected_move_pct": net_expected,
        "catalysts": catalysts,
        "news_available": senti.get("headline_count", 0) > 0,
    }


def _notes(profile: dict[str, Any], scan: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    typ = profile.get("typical_catalyst_move_pct")
    idio = profile.get("idiosyncratic_pct")
    if typ is not None:
        notes.append(
            f"A material stock-specific catalyst has historically moved this "
            f"name ~{typ:.1f}% on the day (90th-pct abnormal return)."
        )
    if idio is not None:
        notes.append(
            f"{idio:.0f}% of its moves are stock-specific (not macro) — "
            f"{'highly' if idio >= 60 else 'moderately' if idio >= 35 else 'mildly'} "
            f"news-sensitive."
        )
    if scan.get("n_catalysts"):
        notes.append(
            f"{scan['n_catalysts']} project/catalyst headline(s) in the recent "
            f"flow — net tone {scan['net_tone']}, implied ~{scan['net_expected_move_pct']:+.1f}%."
        )
    elif not scan.get("news_available"):
        notes.append("No live news available (offline) — showing reaction profile only.")
    else:
        notes.append("No project/catalyst headlines in the recent news flow.")
    return notes


def project_impact(user_ticker: str, limit: int = 25) -> dict[str, Any]:
    """Estimate how future-project news may move a stock, via event study.

    Args:
        user_ticker: EGX code or nickname.
        limit: Max headlines per language to scan. Default 25.
    """
    canonical, _yahoo, name = resolve_ticker(user_ticker)
    cache = price_cache._load() or {}
    drivers = cache.get("drivers", {}).get(canonical)
    if not drivers:
        return {
            "ticker": canonical, "name": name,
            "error": "no cached drivers — run refresh_price_cache first",
        }

    profile = _reaction_profile(canonical, drivers)
    if "error" in profile:
        return {"ticker": canonical, "name": name, **profile}

    scan = _scan_catalysts(canonical, profile, limit)

    return {
        "ticker": canonical,
        "name": name,
        "as_of": datetime.utcnow().isoformat() + "Z",
        "reaction_profile": profile,
        "catalyst_scan": scan,
        "interpretation": _notes(profile, scan),
        "method": (
            "Event study on cached prices: abnormal return = actual daily "
            "return − factor-model expectation (alpha + betas on EGX30/EGP/"
            "Brent/gold/EM). The abnormal-return distribution calibrates the "
            "typical stock-specific catalyst move; recent project headlines "
            "set direction via the sentiment lexicon."
        ),
        "disclaimer": (
            "Statistical reaction estimate, not a price forecast. Magnitude is "
            "this stock's historical typical move on stock-specific news; "
            "direction is headline tone. Size and confirm before acting."
        ),
    }
