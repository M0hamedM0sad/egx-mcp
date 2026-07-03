"""Daily EGX pre-market briefing — generates an HTML report and optionally
emails it. Designed to run from Windows Task Scheduler at 09:00 Cairo time
on EGX trading days (Sun-Thu).

Output:
  1. briefings/briefing_YYYY-MM-DD.html — always written
  2. Email delivery — if SMTP env vars are set:
       SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, BRIEFING_EMAIL_TO

Run manually:
  python -m scripts.daily_briefing

Skip the day check (force run on weekend for testing):
  python -m scripts.daily_briefing --force

Sections in the briefing:
  • Date, EGX session status (open/closed today)
  • Macro snapshot — EGP/USD, Brent, CBE rates, regime
  • Today's W1 weekly top picks with sized levels
  • V8b monthly verdicts on top picks (multi-horizon view)
  • Portfolio summary — current P&L
  • Portfolio vs market — YTD return comparison
  • Catalyst alerts — any picks with earnings within 7 days
  • Honest disclaimer
"""
from __future__ import annotations

import io
import json
import os
import smtplib
import sys
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

# Force UTF-8 stdout/stderr for Windows Task Scheduler logs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from egx_mcp.data import (
    weekly, decision, macro, portfolio, regime,
    egx_listing, market, risk_free, sizing,
    technicals, fundamentals, news, sentiment, news_filter,
    debate as debate_mod, reflection,
    tv_scraper, egx_official,
)
from egx_mcp.data import calendar as cal_mod


# ---- Trade plan parameters (override via env vars if you want) ----
PORTFOLIO_NAV_EGP = float(os.environ.get("EGX_PORTFOLIO_NAV_EGP", "500000"))
RISK_PCT_PER_TRADE = float(os.environ.get("EGX_RISK_PCT_PER_TRADE", "1.0"))
ATR_STOP_MULTIPLE = float(os.environ.get("EGX_ATR_STOP_MULTIPLE", "2.0"))
W1_HOLD_DAYS = 5  # horizon the W1 model is validated for
LIMIT_BUFFER_PCT = 0.5  # max 0.5% above last close for entry limit


def build_trade_plan(ticker: str, last_price: float | None) -> dict:
    """Build a complete trade plan for one W1 pick using ATR-based sizing.

    Returns:
        action          BUY at next open
        entry_limit     Last close × (1 + 0.5%) — cap to avoid chasing
        stop_loss       2 × ATR below entry (Van Tharp / turtle convention)
        target          1:2 R/R — 4 × ATR above entry
        shares          Sized so that hitting stop = 1% of NAV
        position_cost   Shares × entry, in EGP
        position_pct    Position weight as % of NAV
        risk_egp        Max EGP loss if stop fills
        time_stop       Force-close date — 5 trading days from now
        sell_when       Plain-English exit rules
    """
    try:
        s = sizing.position_size(
            ticker,
            portfolio_value_egp=PORTFOLIO_NAV_EGP,
            risk_pct=RISK_PCT_PER_TRADE,
            atr_multiple=ATR_STOP_MULTIPLE,
            max_position_pct=10.0,
        )
    except Exception as e:
        return {"error": str(e)}

    if "error" in s:
        return {"error": s["error"]}

    px = s.get("price")
    entry_limit = round(px * (1 + LIMIT_BUFFER_PCT / 100), 2) if px else None

    # Calendar-day proxy for time stop (5 trading days ≈ 7 calendar days)
    time_stop_date = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")

    return {
        "action": "BUY at open with limit",
        "entry_limit_egp": entry_limit,
        "last_close_egp": round(px, 4) if px else None,
        "atr_14": s.get("atr_14"),
        "stop_loss_egp": s.get("stop_loss_price"),
        "stop_distance_egp": s.get("stop_distance_egp"),
        "scale_out_egp": s.get("scale_out_price"),
        "scale_out_shares": s.get("scale_out_shares"),
        "target_egp": s.get("target_price"),
        "runner_shares": s.get("runner_shares"),
        "shares": s.get("shares"),
        "position_cost_egp": s.get("position_cost_egp"),
        "position_pct_of_nav": s.get("position_weight_pct"),
        "risk_egp": s.get("risk_egp"),
        "reward_to_risk": s.get("reward_to_risk"),
        "time_stop_date": time_stop_date,
        "expected_hold_days": W1_HOLD_DAYS,
        "sell_when": [
            f"Scale-out: at {s.get('scale_out_price')} EGP, sell {s.get('scale_out_shares')} shares (half) and move stop to breakeven on the rest",
            f"Target: at {s.get('target_price')} EGP, sell the remaining {s.get('runner_shares')} shares — OR trail by 1×ATR ({s.get('atr_14')}) on new highs to let it run",
            f"Stop {s.get('stop_loss_price')} EGP hit — exit immediately",
            f"Time stop {time_stop_date}: applies ONLY if scale-out never triggered (full position still at risk). Once scale-out hits, the runner trails freely.",
            "If pick re-enters W1 top-5 next Sunday — extend the trail, otherwise rotate",
        ],
        "capped_by_liquidity": s.get("capped_by_liquidity"),
        "capped_by_max_position": s.get("capped_by_max_position"),
    }


# ---------------------------------------------------------------------------
# Per-pick analyst summaries
# ---------------------------------------------------------------------------

def technical_summary(ticker: str) -> dict:
    """Pull technicals + condense the indicator block into a one-line read."""
    try:
        t = technicals.compute(ticker, period="6mo")
    except Exception as e:
        return {"error": str(e)[:80]}
    if "error" in t:
        return {"error": t["error"]}
    ind = t.get("indicators") or {}
    rsi = ind.get("rsi_14")
    macd = ind.get("macd")
    macd_sig = ind.get("macd_signal")
    sma50 = ind.get("sma_50")
    sma200 = ind.get("sma_200")
    bb_u = ind.get("bb_upper")
    bb_l = ind.get("bb_lower")
    bb_m = ind.get("bb_middle")
    atr = ind.get("atr_14")
    px = t.get("price")

    bullets: list[str] = []
    if rsi is not None:
        if rsi >= 70:
            bullets.append(f"RSI {rsi:.0f} — overbought")
        elif rsi <= 30:
            bullets.append(f"RSI {rsi:.0f} — oversold")
        elif 50 <= rsi < 70:
            bullets.append(f"RSI {rsi:.0f} — healthy uptrend")
        elif 30 < rsi < 50:
            bullets.append(f"RSI {rsi:.0f} — soft / mean-revert zone")
    if macd is not None and macd_sig is not None:
        bullets.append(f"MACD {'bullish' if macd > macd_sig else 'bearish'} ({macd - macd_sig:+.3f})")
    if sma50 and sma200:
        if sma50 > sma200:
            bullets.append(f"MA50 > MA200 — golden-cross regime")
        else:
            bullets.append(f"MA50 < MA200 — death-cross regime")
    if px and bb_u and bb_l:
        # Position within Bollinger band, 0 = lower, 1 = upper
        rng = bb_u - bb_l
        if rng > 0:
            pos = (px - bb_l) / rng
            if pos >= 0.95:
                bullets.append("Riding upper Bollinger — extended")
            elif pos <= 0.05:
                bullets.append("Hugging lower Bollinger — compressed")
    if atr and px:
        bullets.append(f"ATR {atr:.2f} ({atr/px*100:.1f}% of px) — sizing input")

    return {
        "rsi_14": rsi,
        "macd_minus_signal": (macd - macd_sig) if (macd is not None and macd_sig is not None) else None,
        "sma_50": sma50,
        "sma_200": sma200,
        "atr_14": atr,
        "atr_pct_of_price": round(atr / px * 100, 2) if (atr and px) else None,
        "bullets": bullets,
        "signals": t.get("signals"),
    }


def fundamental_summary(ticker: str) -> dict:
    """Pull fundamentals + emit a one-line read with sector context."""
    try:
        f = fundamentals.get_fundamentals(ticker)
    except Exception as e:
        return {"error": str(e)[:80]}
    if "error" in f:
        return {"error": f["error"]}

    pe = f.get("pe_ratio")
    pb = f.get("pb_ratio")
    roe = f.get("roe_pct")
    margin = f.get("profit_margin_pct")
    de = f.get("debt_to_equity")
    div_y = f.get("dividend_yield_pct")

    bullets: list[str] = []
    if pe is not None:
        if pe < 8:
            bullets.append(f"P/E {pe:.1f} — cheap")
        elif pe < 15:
            bullets.append(f"P/E {pe:.1f} — fair")
        else:
            bullets.append(f"P/E {pe:.1f} — premium")
    if pb is not None:
        bullets.append(f"P/B {pb:.2f}")
    if roe is not None:
        if roe >= 20:
            bullets.append(f"ROE {roe:.1f}% — strong")
        elif roe >= 10:
            bullets.append(f"ROE {roe:.1f}% — adequate")
        else:
            bullets.append(f"ROE {roe:.1f}% — weak")
    if margin is not None:
        bullets.append(f"Margin {margin:.1f}%")
    if de is not None:
        if de >= 1.5:
            bullets.append(f"D/E {de:.2f} — highly levered")
        else:
            bullets.append(f"D/E {de:.2f}")
    if div_y is not None and div_y > 0:
        bullets.append(f"Yield {div_y:.1f}%")
    if f.get("pe_was_corrected"):
        bullets.append("(P/E auto-corrected from Yahoo's bogus value)")

    return {
        "pe_ratio": pe,
        "pb_ratio": pb,
        "roe_pct": roe,
        "profit_margin_pct": margin,
        "debt_to_equity": de,
        "dividend_yield_pct": div_y,
        "bullets": bullets,
        "pe_was_corrected": f.get("pe_was_corrected"),
    }


def market_news_summary(
    limit_per_lang: int = 12,
    portfolio_csv: str | None = None,
    watchlist_fallback: list[str] | None = None,
) -> dict:
    """Pull market-wide news (EN + AR), filter to headlines that impact
    the user's portfolio (or a watchlist when portfolio is empty), and
    add a sentiment read on the filtered set.

    When the portfolio CSV has no positions, falls back to filtering
    against the `watchlist_fallback` tickers (typically today's W1
    picks). This way a flat-cash user still gets a focused briefing
    rather than an unfiltered firehose.
    """
    raw: list[dict] = []

    try:
        en = news.fetch(ticker=None, lang="en", limit=limit_per_lang)
        for art in (en.get("articles") or [])[:limit_per_lang]:
            if art.get("title"):
                raw.append({**art, "lang": "en"})
    except Exception as e:
        log.warning(f"market EN news failed: {e}")

    try:
        ar = news.fetch(ticker=None, lang="ar", limit=limit_per_lang)
        for art in (ar.get("articles") or [])[:limit_per_lang]:
            if art.get("title"):
                raw.append({**art, "lang": "ar"})
    except Exception as e:
        log.warning(f"market AR news failed: {e}")

    # Decide which ticker set drives the filter
    held_tickers: list[str] = []
    if portfolio_csv:
        try:
            from egx_mcp.data import portfolio as port_mod
            summary = port_mod.summary(csv_path=portfolio_csv)
            if "error" not in summary:
                held_tickers = [p["ticker"].upper() for p in (summary.get("positions") or [])]
        except Exception as e:
            log.warning(f"portfolio load failed: {e}")

    filter_basis = "portfolio"
    target_tickers = held_tickers
    if not target_tickers and watchlist_fallback:
        target_tickers = [t.upper() for t in watchlist_fallback]
        filter_basis = "watchlist"

    if target_tickers:
        try:
            filtered = news_filter.filter_for_portfolio(
                raw,
                target_tickers,
                include_sector_matches=True,
                include_macro=False,
            )
        except Exception as e:
            log.warning(f"news_filter failed, returning raw: {e}")
            filtered = raw
            filter_basis = "unfiltered (filter error)"
    else:
        filtered = raw
        filter_basis = "unfiltered (no portfolio, no watchlist)"

    # Sentiment scored on the FILTERED set so the tone reflects what
    # the user actually cares about.
    titles = " || ".join(h.get("title", "") for h in filtered)
    sent_payload = None
    try:
        score, _ = sentiment._score_text(titles, "en")
        score_ar, _ = sentiment._score_text(titles, "ar")
        # Average the two channels; either may be 0 if no terms hit
        active = [s for s in (score, score_ar) if s != 0]
        agg = sum(active) / len(active) if active else 0.0
        sent_payload = {
            "label": sentiment._label(agg),
            "aggregate_score": round(agg, 3),
            "headline_count": len(filtered),
            "method": "lexicon scored over portfolio-filtered headlines",
        }
    except Exception as e:
        sent_payload = {"error": str(e)[:80]}

    return {
        "headlines": filtered,
        "raw_headline_count": len(raw),
        "filtered_headline_count": len(filtered),
        "filter_basis": filter_basis,
        "filter_target_tickers": target_tickers,
        "sentiment": sent_payload,
        "filter_method": (
            "Direct ticker-name match wins. Sector keyword match falls "
            "back to 'sector' impact level. Macro headlines (CBE, EGP, "
            "rates) are not included by default — they affect everyone."
        ),
    }


def per_ticker_news(ticker: str, limit: int = 4) -> dict:
    """Pull a few headlines for a single name. EN preferred, AR fallback."""
    out: list[dict] = []
    for lang in ("en", "ar"):
        try:
            n = news.fetch(ticker=ticker, lang=lang, limit=limit)
            for art in (n.get("articles") or [])[:limit]:
                if art.get("title") and art.get("title") not in {h["title"] for h in out}:
                    out.append({**art, "lang": lang})
                if len(out) >= limit:
                    break
        except Exception:
            continue
        if len(out) >= limit:
            break

    sent = None
    try:
        sent = sentiment.analyze_sentiment(ticker, lang="both", limit=limit)
    except Exception:
        pass

    return {"headlines": out[:limit], "sentiment": sent}


# ---------------------------------------------------------------------------
# Position-aware verdict translator
# ---------------------------------------------------------------------------
#
# The model verdicts (BUY/ACCUMULATE/HOLD/REDUCE/AVOID/WAIT) implicitly
# assume the reader might already hold the name. "HOLD" on a stock
# you don't own is meaningless — you can't hold what you don't have.
# Same for REDUCE / TRIM. This translator makes the action explicit
# given whether the name is in the portfolio CSV.
#
#   HELD                              NOT HELD
#   --------------------              --------------------
#   BUY/ACCUMULATE  → ADD             BUY/ACCUMULATE → BUY
#   HOLD            → HOLD            HOLD           → WATCH
#   REDUCE          → TRIM            REDUCE         → SKIP
#   AVOID           → EXIT            AVOID          → SKIP
#   WAIT            → HOLD            WAIT           → WATCH

_HELD_MAP = {
    "BUY":        "ADD",
    "ACCUMULATE": "ADD",
    "HOLD":       "HOLD",
    "REDUCE":     "TRIM",
    "AVOID":      "EXIT",
    "WAIT":       "HOLD",
}
_NOT_HELD_MAP = {
    "BUY":        "BUY",
    "ACCUMULATE": "BUY",
    "HOLD":       "WATCH",
    "REDUCE":     "SKIP",
    "AVOID":      "SKIP",
    "WAIT":       "WATCH",
}


def position_aware_action(verdict: str | None, is_held: bool) -> str:
    """Translate a model verdict into a position-aware action label."""
    if not verdict:
        return "—"
    table = _HELD_MAP if is_held else _NOT_HELD_MAP
    return table.get(verdict.upper(), verdict)


def held_set_from_portfolio(b: dict) -> set[str]:
    """Pull the set of held tickers from the briefing's portfolio block."""
    pf = b.get("portfolio") or {}
    if not isinstance(pf, dict):
        return set()
    return {p["ticker"].upper() for p in (pf.get("positions") or []) if p.get("ticker")}


CAIRO_TZ_OFFSET = 2  # UTC+2 standard time; +3 during DST (Apr-Oct)
EGX_TRADING_DAYS = {6, 0, 1, 2, 3}  # Sun=6, Mon=0, ... Thu=3 in Python's weekday()
# Python weekday: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
# EGX trades Sun-Thu, so allowed weekdays = {6, 0, 1, 2, 3}

OUTPUT_DIR = Path(__file__).parent.parent / "briefings"
OUTPUT_DIR.mkdir(exist_ok=True)


def is_trading_day(d: datetime) -> bool:
    return d.weekday() in EGX_TRADING_DAYS


def get_egx30_basket_ytd() -> tuple[float | None, str]:
    """Compute YTD return of the 13-name liquid EGX basket as the market benchmark."""
    proxy = ["COMI", "HDBK", "CIRA", "SWDY", "ETEL", "ABUK", "EFID",
             "TMGH", "ORWE", "FWRY", "EAST", "MFPC", "PHAR"]
    year_start = f"{datetime.utcnow().year}-01-01"
    rets = []
    for tk in proxy:
        try:
            from egx_mcp.data.universe import resolve_ticker
            _, yahoo, _ = resolve_ticker(tk)
            h = yf.Ticker(yahoo).history(start=year_start, interval="1d")
            if h is None or h.empty or len(h) < 2:
                continue
            r = float(h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100
            rets.append(r)
        except Exception:
            continue
    if not rets:
        return None, "Market YTD: insufficient data"
    avg = sum(rets) / len(rets)
    return avg, f"Market YTD ({len(rets)}-name liquid basket EW): {avg:+.2f}%"


def build_briefing(force: bool = False) -> dict:
    """Build the full briefing data structure."""
    now = datetime.utcnow()
    cairo = now + timedelta(hours=CAIRO_TZ_OFFSET)
    is_egx_day = is_trading_day(cairo)

    if not is_egx_day and not force:
        return {"skip": True, "reason": f"{cairo.strftime('%A')} is not an EGX trading day"}

    out = {
        "as_of_utc": now.isoformat() + "Z",
        "cairo_local": cairo.strftime("%A %Y-%m-%d %H:%M"),
        "is_trading_day": is_egx_day,
    }

    # Macro
    try:
        out["macro"] = macro.get_context()
    except Exception as e:
        out["macro"] = {"error": str(e)}

    # EGX index spot (30 + 70) — multi-source chain
    #   1. egx_official (Playwright vs egx.com.eg) — best-effort, TSPD-fragile
    #   2. TradingView — reliable for EGX 30, no coverage for EGX 70
    #   3. yfinance — final fallback, unreliable for EGX symbols
    out["egx30_spot"] = None
    out["egx70_spot"] = None
    out["egx_index_sources"] = []

    try:
        official = egx_official.fetch_indices()
        if official and "error" not in official and (official.get("indices") or {}):
            idx = official["indices"]
            for k, v in idx.items():
                if "EGX30" in k.upper() and "TR" not in k.upper() and out["egx30_spot"] is None:
                    out["egx30_spot"] = {
                        "value": v.get("value"), "change_pct": v.get("change_pct"),
                        "source": "egx.com.eg",
                    }
                if ("EGX70" in k.upper() or "EGX 70" in (v.get("label") or "").upper()) and out["egx70_spot"] is None:
                    out["egx70_spot"] = {
                        "value": v.get("value"), "change_pct": v.get("change_pct"),
                        "source": "egx.com.eg",
                    }
            out["egx_index_sources"].append(f"egx.com.eg ({len(idx)} indices)")
    except Exception as e:
        out["egx_index_sources"].append(f"egx.com.eg failed: {str(e)[:60]}")

    if out["egx30_spot"] is None:
        try:
            tv = tv_scraper.fetch_egx30()
            if tv and "error" not in tv and tv.get("price") is not None:
                out["egx30_spot"] = {
                    "value": tv["price"], "change_pct": None, "source": "tradingview.com",
                    "chart_url": tv.get("chart_url"),
                }
                out["egx_index_sources"].append("tradingview.com (EGX 30)")
        except Exception as e:
            out["egx_index_sources"].append(f"tradingview failed: {str(e)[:60]}")

    if out["egx30_spot"] is None:
        try:
            yf_30 = market.get_index("EGX30")
            out["egx30_spot"] = {**yf_30, "source": "yfinance"}
        except Exception as e:
            out["egx30_spot"] = {"error": str(e)[:100], "source": "yfinance"}

    if out["egx70_spot"] is None:
        try:
            yf_70 = market.get_index("EGX70")
            if yf_70.get("value") is not None:
                out["egx70_spot"] = {**yf_70, "source": "yfinance"}
            else:
                out["egx70_spot"] = {"error": "EGX 70 unavailable from any source today",
                                     "source": "yfinance (failed)"}
        except Exception as e:
            out["egx70_spot"] = {"error": str(e)[:100], "source": "yfinance"}

    # Gold prices in EGP — 24K / 21K / 18K per gram + Egyptian pound
    try:
        out["gold_egp"] = macro.gold_prices_egp()
    except Exception as e:
        out["gold_egp"] = {"error": str(e)[:100]}

    # Regime
    try:
        out["regime"] = regime.classify()
    except Exception as e:
        out["regime"] = {"error": str(e)}

    # Risk-free rate
    try:
        out["rf_rate"] = risk_free.get_rate()
    except Exception as e:
        out["rf_rate"] = {"error": str(e)}

    # W1 weekly top picks
    try:
        out["w1_picks"] = weekly.rank_universe(top_n=5)
    except Exception as e:
        out["w1_picks"] = {"error": str(e)}

    # Ex-dividend pre-filter — drop any top pick going ex-div within 3
    # trading sessions and promote from runners_up. The ex-div price gap
    # mechanically eats into the 5d target and skewed the 2026-05-02 cohort
    # (POUL and CCAP both had record dates that week).
    out["exdiv_filtered"] = []
    try:
        picks_list = (out["w1_picks"].get("top_picks") or []) if isinstance(out["w1_picks"], dict) else []
        runners = (out["w1_picks"].get("runners_up") or []) if isinstance(out["w1_picks"], dict) else []
        kept: list[dict] = []
        for p in picks_list:
            tk = p["ticker"]
            try:
                c = cal_mod.get_calendar(tk)
            except Exception:
                kept.append(p)
                continue
            d2x = c.get("days_to_exdividend")
            # 3 trading sessions ≈ 4 calendar days on EGX (Sun-Thu schedule).
            if d2x is not None and 0 <= d2x <= 4:
                out["exdiv_filtered"].append({
                    "ticker": tk,
                    "ex_dividend_date": c.get("ex_dividend_date"),
                    "days_to_exdividend": d2x,
                    "reason": "within 3 trading sessions of ex-div — target gets eaten by the gap",
                })
            else:
                kept.append(p)
        # Backfill from runners_up so we still output top 5
        while len(kept) < 5 and runners:
            cand = runners.pop(0)
            tk = cand["ticker"]
            try:
                c = cal_mod.get_calendar(tk)
                d2x = c.get("days_to_exdividend")
                if d2x is not None and 0 <= d2x <= 4:
                    out["exdiv_filtered"].append({
                        "ticker": tk,
                        "ex_dividend_date": c.get("ex_dividend_date"),
                        "days_to_exdividend": d2x,
                        "reason": "runner-up also within 3 sessions of ex-div",
                    })
                    continue
            except Exception:
                pass
            kept.append(cand)
        if isinstance(out["w1_picks"], dict):
            out["w1_picks"]["top_picks"] = kept
            out["w1_picks"]["runners_up"] = runners
    except Exception as e:
        out["exdiv_filter_error"] = str(e)[:120]

    # V8b monthly verdicts on the W1 picks (cross-check) PLUS the whole
    # audited-fundamentals universe. Deciding only the day's momentum picks
    # gave the learning loop 5 calls/day drawn from a biased (momentum-
    # selected) sample; grading the full covered list accumulates evidence
    # ~6x faster and lets learn.py fit thresholds/weights on the same
    # population decide() actually serves.
    try:
        verdicts = []
        decided: set[str] = set()
        pick_tickers = [p["ticker"] for p in (out["w1_picks"].get("top_picks", []) or [])]
        audited: list[str] = []
        try:
            import csv as _csv
            _audited_csv = Path(__file__).parent.parent / "egx_fundamentals_audited.csv"
            with _audited_csv.open(encoding="utf-8-sig") as f:
                audited = [row["ticker"].strip() for row in _csv.DictReader(f)
                           if row.get("ticker", "").strip()]
        except Exception as e:
            out["v8b_universe_error"] = str(e)[:120]
        for tk in pick_tickers + [t for t in audited if t not in pick_tickers]:
            if tk in decided:
                continue
            decided.add(tk)
            try:
                d = decision.decide(tk)
                verdicts.append({
                    "ticker": tk,
                    "v8b_verdict": d.get("verdict"),
                    "v8b_score": d.get("composite_score"),
                    "v8b_conviction": d.get("conviction"),
                    "blocking_catalysts": d.get("blocking_catalysts", []),
                    # Per-decision sub-scores — recorded so the learning loop can
                    # later analyze which factors actually drive forward excess.
                    "v8b_subscores": d.get("subscores"),
                })
            except Exception as e:
                verdicts.append({"ticker": tk, "error": str(e)[:80]})
        out["v8b_verdicts"] = verdicts
    except Exception as e:
        out["v8b_verdicts"] = {"error": str(e)}

    # Chairman read per W1 pick — agentic verdict that DOES use sentiment
    # + bull/bear theses. Compared to V8b in the rendered briefing so
    # news-driven dissent is visible. Each verdict is logged to the
    # decision JSONL so reflect_on_decisions can grade them later.
    out["chairman_per_pick"] = {}
    out["divergences"] = []
    for p in (out["w1_picks"].get("top_picks", []) or []):
        tk = p["ticker"]
        try:
            d = debate_mod.debate(tk, include_sentiment=True)
        except Exception as e:
            out["chairman_per_pick"][tk] = {"error": str(e)[:80]}
            continue
        if d.get("error"):
            out["chairman_per_pick"][tk] = {"error": d["error"]}
            continue

        chair = d.get("chairman") or {}
        sent = d.get("sentiment_summary") or {}
        out["chairman_per_pick"][tk] = {
            "verdict": chair.get("verdict"),
            "conviction": chair.get("conviction"),
            "edge": chair.get("edge"),
            "rationale": chair.get("rationale"),
            "deciding_factors": chair.get("deciding_factors", []),
            "sentiment_label": sent.get("label"),
            "sentiment_score": sent.get("aggregate_score"),
        }

        # Compare against V8b — flag when news-aware view differs
        v8b_match = next((v for v in (out["v8b_verdicts"] or [])
                          if isinstance(v, dict) and v.get("ticker") == tk), None)
        v8b_verdict = (v8b_match or {}).get("v8b_verdict")
        if v8b_verdict and chair.get("verdict") and v8b_verdict != chair.get("verdict"):
            out["divergences"].append({
                "ticker": tk,
                "v8b": v8b_verdict,
                "chairman": chair["verdict"],
                "rationale": chair.get("rationale"),
                "sentiment": sent.get("label"),
            })

        # Append to decision log for forward grading
        try:
            reflection.log_decision(
                ticker=tk,
                verdict=chair.get("verdict") or "HOLD",
                conviction=chair.get("conviction"),
                composite_score=d.get("composite_score"),
                tags=["chairman", "daily_briefing", f"sentiment_{sent.get('label','neutral')}"],
                note=(chair.get("rationale") or "")[:160],
            )
        except Exception:
            pass

    # Trade plans for each W1 pick + live tape prices to override the
    # split-adjusted historical close that the W1 ranker reports.
    out["trade_plans"] = {}
    for p in (out["w1_picks"].get("top_picks", []) or []):
        tk = p["ticker"]
        out["trade_plans"][tk] = build_trade_plan(tk, p.get("price"))
        # Pull the live tape price so the displayed price matches the order ticket
        try:
            q = market.get_quote(tk)
            live_px = q.get("price")
            if live_px is not None and p.get("price") is not None:
                ratio = live_px / p["price"]
                # Flag if live and adjusted differ by >25% — likely a split
                if abs(ratio - 1) > 0.25:
                    p["adjusted_close_egp"] = p["price"]
                    p["price"] = live_px
                    p["split_warning"] = (
                        f"Live tape {live_px} differs from adjusted close "
                        f"{p['adjusted_close_egp']} — likely recent split. Use trade plan."
                    )
                else:
                    p["price"] = live_px
        except Exception:
            pass
    # Per-pick technical + fundamental + news summaries
    out["technicals_per_pick"] = {}
    out["fundamentals_per_pick"] = {}
    out["news_per_pick"] = {}
    for p in (out["w1_picks"].get("top_picks", []) or []):
        tk = p["ticker"]
        out["technicals_per_pick"][tk] = technical_summary(tk)
        out["fundamentals_per_pick"][tk] = fundamental_summary(tk)
        out["news_per_pick"][tk] = per_ticker_news(tk, limit=3)

    # Market-wide news + sentiment.
    # Filtered by holdings if portfolio CSV is set + populated, else by
    # today's W1 picks so a flat-cash user still gets a focused briefing.
    w1_tickers = [p["ticker"] for p in (out["w1_picks"].get("top_picks", []) or [])]
    try:
        out["market_news"] = market_news_summary(
            limit_per_lang=12,
            portfolio_csv=os.environ.get("EGX_PORTFOLIO_CSV"),
            watchlist_fallback=w1_tickers,
        )
    except Exception as e:
        out["market_news"] = {"error": str(e)[:120]}

    out["trade_plan_params"] = {
        "portfolio_nav_egp": PORTFOLIO_NAV_EGP,
        "risk_pct_per_trade": RISK_PCT_PER_TRADE,
        "atr_stop_multiple": ATR_STOP_MULTIPLE,
        "expected_hold_days": W1_HOLD_DAYS,
    }

    # Catalyst alerts on the W1 picks
    try:
        catalysts = []
        for p in (out["w1_picks"].get("top_picks", []) or []):
            tk = p["ticker"]
            try:
                c = cal_mod.get_calendar(tk)
                d2e = c.get("days_to_earnings")
                if d2e is not None and 0 <= d2e <= 7:
                    catalysts.append({
                        "ticker": tk,
                        "type": "earnings",
                        "days": d2e,
                        "date": c.get("next_earnings_date"),
                    })
                d2x = c.get("days_to_exdividend")
                if d2x is not None and 0 <= d2x <= 5:
                    catalysts.append({
                        "ticker": tk,
                        "type": "ex-dividend",
                        "days": d2x,
                        "date": c.get("ex_dividend_date"),
                    })
            except Exception:
                continue
        out["catalysts"] = catalysts
    except Exception as e:
        out["catalysts"] = {"error": str(e)}

    # Portfolio
    try:
        out["portfolio"] = portfolio.summary()
    except Exception as e:
        out["portfolio"] = {"error": str(e)}

    # Market YTD comparison
    try:
        market_ytd, market_label = get_egx30_basket_ytd()
        out["market_ytd_pct"] = market_ytd
        out["market_label"] = market_label
        if isinstance(out.get("portfolio"), dict) and out["portfolio"].get("total_pnl_pct") is not None and market_ytd is not None:
            out["portfolio_vs_market_pp"] = out["portfolio"]["total_pnl_pct"] - market_ytd
    except Exception as e:
        out["market_ytd_pct"] = None
        out["market_label"] = f"Error: {e}"

    return out


def render_html(b: dict) -> str:
    """Render the briefing dict as a formatted HTML email body."""
    if b.get("skip"):
        return f"""<html><body><p>Skipped: {b.get("reason")}</p></body></html>"""

    macro = b.get("macro", {}) or {}
    regime_d = b.get("regime", {}) or {}
    rf = b.get("rf_rate", {}) or {}
    w1 = b.get("w1_picks", {}) or {}
    picks = w1.get("top_picks", []) or []
    runners = w1.get("runners_up", []) or []
    verdicts = b.get("v8b_verdicts", []) or []
    catalysts = b.get("catalysts", []) or []
    pf = b.get("portfolio", {}) or {}
    pf_positions = pf.get("positions", []) if isinstance(pf, dict) else []

    # Map verdicts by ticker
    verdict_map = {v["ticker"]: v for v in verdicts if isinstance(v, dict) and "ticker" in v}
    cat_map = {}
    for c in catalysts:
        if isinstance(c, dict) and "ticker" in c:
            cat_map.setdefault(c["ticker"], []).append(c)

    egp = (macro.get("egp_usd") or {}).get("value")
    egp_chg = (macro.get("egp_usd") or {}).get("change_pct")
    brent = (macro.get("brent_usd") or {}).get("value")
    brent_chg = (macro.get("brent_usd") or {}).get("change_pct")
    cbe = (macro.get("cbe_rates") or {}).get("midpoint_pct")
    regime_label = regime_d.get("regime", "UNKNOWN")
    regime_desc = regime_d.get("description", "")

    css = """
    <style>
    body { font-family: -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
           color: #1a1a1a; max-width: 900px; margin: 24px auto; padding: 0 16px;
           font-size: 14px; line-height: 1.5; }
    h1 { color: #0d3e66; border-bottom: 2px solid #0d3e66; padding-bottom: 6px; }
    h2 { color: #0d3e66; margin-top: 24px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; }
    th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; font-size: 13px; }
    th { background: #f4f6f9; color: #0d3e66; font-weight: 600; }
    tr:hover { background: #fafbfc; }
    .pos { color: #0a7d3a; font-weight: 600; }
    .neg { color: #c0392b; font-weight: 600; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 10px;
            font-size: 11px; font-weight: 600; }
    .buy   { background: #d4edda; color: #155724; }
    .accum { background: #cce5ff; color: #004085; }
    .hold  { background: #fff3cd; color: #856404; }
    .reduce, .avoid { background: #f8d7da; color: #721c24; }
    .alert { background: #fff3cd; padding: 10px; border-left: 4px solid #ffc107;
             margin: 10px 0; border-radius: 4px; }
    .summary-box { background: #f4f6f9; padding: 12px; border-radius: 4px;
                   margin: 12px 0; }
    .footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid #ddd;
              font-size: 11px; color: #666; }
    </style>
    """

    parts = [f"<html><head>{css}</head><body>"]
    parts.append(f"<h1>EGX Pre-Market Briefing</h1>")
    parts.append(f"<p><b>{b['cairo_local']}</b> | EGX opens 10:00 Cairo</p>")

    # MACRO
    parts.append("<h2>Macro Snapshot</h2>")
    parts.append("<div class='summary-box'>")
    parts.append("<table>")
    parts.append(f"<tr><th>Indicator</th><th>Level</th><th>Change</th><th>Note</th></tr>")

    # EGX index spot — pulled out front so the numbers people glance at first
    egx30 = b.get("egx30_spot") or {}
    if not egx30.get("error") and egx30.get("value") is not None:
        v = egx30.get("value")
        ch = egx30.get("change_pct")
        cls = "pos" if (ch or 0) > 0 else "neg"
        chg_str = f"{ch:+.2f}%" if ch is not None else "—"
        src = egx30.get("source", "?")
        chart = egx30.get("chart_url")
        chart_html = f" <a href='{chart}' style='font-size:11px;color:#666'>chart</a>" if chart else ""
        parts.append(f"<tr><td>EGX 30</td><td><b>{v:,.0f}</b> pts</td>"
                     f"<td class='{cls}'>{chg_str}</td>"
                     f"<td>Flagship index — market beta <span style='color:#888;font-size:11px'>[{src}]</span>{chart_html}</td></tr>")
    egx70 = b.get("egx70_spot") or {}
    if not egx70.get("error") and egx70.get("value") is not None:
        v = egx70.get("value")
        ch = egx70.get("change_pct")
        cls = "pos" if (ch or 0) > 0 else "neg"
        chg_str = f"{ch:+.2f}%" if ch is not None else "—"
        src = egx70.get("source", "?")
        parts.append(f"<tr><td>EGX 70</td><td><b>{v:,.0f}</b> pts</td>"
                     f"<td class='{cls}'>{chg_str}</td>"
                     f"<td>Mid/small-cap breadth <span style='color:#888;font-size:11px'>[{src}]</span></td></tr>")
    elif egx70.get("error"):
        parts.append(f"<tr><td>EGX 70</td><td colspan='3' style='color:#888;font-size:12px'>"
                     f"Unavailable today — {egx70.get('error')[:80]} "
                     f"(no clean public source after the egx.com.eg / yfinance fallbacks)</td></tr>")

    if egp is not None:
        cls = "pos" if (egp_chg or 0) > 0 else "neg"
        chg_str = f"{egp_chg:+.2f}%" if egp_chg is not None else "—"
        parts.append(f"<tr><td>USD/EGP</td><td>{egp}</td><td class='{cls}'>{chg_str}</td><td>EGP weakening = exporter tailwind</td></tr>")
    if brent is not None:
        cls = "pos" if (brent_chg or 0) > 0 else "neg"
        chg_str = f"{brent_chg:+.2f}%" if brent_chg is not None else "—"
        parts.append(f"<tr><td>Brent crude</td><td>${brent:.2f}</td><td class='{cls}'>{chg_str}</td><td>Petrochems / Suez exposure</td></tr>")

    # Gold prices — 24K / 21K / 18K per gram + Egyptian gold pound
    gold = b.get("gold_egp") or {}
    if not gold.get("error") and gold.get("egp_per_gram_24k") is not None:
        parts.append(f"<tr><td>Gold 24K (per g)</td><td>{gold.get('egp_per_gram_24k', 0):,.2f} EGP</td><td>—</td><td>Spot-derived; local dealer ≈ +5-15%</td></tr>")
        parts.append(f"<tr><td>Gold 21K (per g)</td><td>{gold.get('egp_per_gram_21k', 0):,.2f} EGP</td><td>—</td><td>Egyptian jewelry standard</td></tr>")
        parts.append(f"<tr><td>Gold 18K (per g)</td><td>{gold.get('egp_per_gram_18k', 0):,.2f} EGP</td><td>—</td><td>Common alternate alloy</td></tr>")
        parts.append(f"<tr><td>Egyptian gold pound</td><td><b>{gold.get('egyptian_gold_pound_egp', 0):,.0f}</b> EGP</td><td>—</td><td>8 g of 21K (الجنيه الذهب)</td></tr>")

    if cbe is not None:
        parts.append(f"<tr><td>CBE midpoint</td><td>{cbe}%</td><td>—</td><td>{'Tight regime' if cbe>=20 else 'Easing'}</td></tr>")
    parts.append(f"<tr><td>EGP T-bill</td><td>{rf.get('rate_pct', '—')}%</td><td>—</td><td>Excess-return baseline</td></tr>")
    parts.append(f"<tr><td>Market regime</td><td><b>{regime_label}</b></td><td>—</td><td>{regime_desc[:80]}</td></tr>")
    parts.append("</table>")
    if macro.get("regime_flags"):
        parts.append("<p><b>Regime flags:</b></p><ul>")
        for f in macro["regime_flags"]:
            parts.append(f"<li>{f}</li>")
        parts.append("</ul>")
    parts.append("</div>")

    # W1 PICKS
    parts.append("<h2>Today's W1 Weekly Picks (5-day horizon)</h2>")
    parts.append("<p style='color:#666;font-size:12px'>Backtest: +107.8% CAGR / 1.64 Sharpe full window, +131% / 2.78 Sharpe holdout OOS. "
                 f"Eligible universe today: <b>{w1.get('n_eligible', '?')}</b> names.</p>")
    if picks:
        parts.append("<table>")
        parts.append("<tr><th>#</th><th>Ticker</th><th>Score</th><th>Price (EGP)</th>"
                     "<th>1d</th><th>5d</th><th>20d</th><th>Trend</th>"
                     "<th>Vol×ADV</th><th>Breakout</th><th>V8b verdict</th><th>Alert</th></tr>")
        for i, p in enumerate(picks, 1):
            tk = p["ticker"]
            v = verdict_map.get(tk, {})
            verdict = v.get("v8b_verdict", "—")
            vcls = "buy" if verdict == "BUY" else "accum" if verdict == "ACCUMULATE" else \
                   "hold" if verdict == "HOLD" else "reduce" if verdict in ("REDUCE", "AVOID") else "hold"
            cats = cat_map.get(tk, [])
            alert = ""
            if cats:
                for c in cats:
                    alert += f"⚠ {c['type']} in {c['days']}d "
            mom1 = p.get('mom_1d_pct') or 0
            mom5 = p.get('mom_5d_pct') or 0
            mom20 = p.get('mom_20d_pct') or 0
            cls1 = "pos" if mom1 > 0 else "neg"
            cls5 = "pos" if mom5 > 0 else "neg"
            cls20 = "pos" if mom20 > 0 else "neg"
            trend = "↑" if p.get("above_ma20") else "↓"
            bo = p.get("breakout_signal", 0)
            bo_str = "↑20d" if bo > 0 else "↓20d" if bo < 0 else "—"
            chart_url = tv_scraper.chart_url(tk)
            parts.append(
                f"<tr><td>{i}</td>"
                f"<td><a href='{chart_url}' style='color:#0d3e66;font-weight:600;text-decoration:none'>{tk}</a> "
                f"<a href='{chart_url}' style='color:#888;font-size:10px'>📈</a></td>"
                f"<td>{p.get('score','?')}</td>"
                f"<td>{p.get('price','—')}</td>"
                f"<td class='{cls1}'>{mom1:+.1f}%</td>"
                f"<td class='{cls5}'>{mom5:+.1f}%</td>"
                f"<td class='{cls20}'>{mom20:+.1f}%</td>"
                f"<td>{trend}</td>"
                f"<td>{p.get('vol_ratio_today','?')}×</td>"
                f"<td>{bo_str}</td>"
                f"<td><span class='pill {vcls}'>{verdict}</span></td>"
                f"<td>{alert}</td></tr>"
            )
        parts.append("</table>")
        if runners:
            parts.append(f"<p style='color:#666;font-size:12px'>Runners-up: " +
                         ", ".join(f"{r['ticker']} ({r.get('score','?')})" for r in runners[:5]) + "</p>")
    else:
        parts.append(f"<p>No eligible picks today. Reason: {w1.get('error', 'universe filter cleared all')}</p>")

    # CATALYST ALERTS
    if catalysts:
        parts.append("<div class='alert'><b>⚠ Catalyst alerts on today's picks:</b><ul>")
        for c in catalysts:
            parts.append(f"<li><b>{c.get('ticker')}</b> — {c.get('type')} in {c.get('days')} days ({c.get('date')})</li>")
        parts.append("</ul><p style='font-size:12px;color:#666'>Consider sizing down or waiting for the print.</p></div>")

    # VERDICT COMPARISON — position-aware actions
    chair_per_pick = b.get("chairman_per_pick") or {}
    divergences = b.get("divergences") or []
    held = held_set_from_portfolio(b)
    if picks and chair_per_pick:
        parts.append("<h2>Action Recommendations</h2>")
        parts.append(
            "<p style='color:#666;font-size:12px'>"
            "Two horizons: <b>5-day</b> = today's W1 weekly trading rank "
            "(top-5 = trade candidates). <b>1-month</b> = chairman synthesis "
            "(bull/bear debate, news + sentiment, valuation, technicals). "
            "Actions are <b>position-aware</b>: held names get ADD/HOLD/TRIM/EXIT, "
            "unheld names get BUY/WATCH/SKIP. They can disagree legitimately — "
            "different time frames, different signals."
            "</p>"
        )
        if divergences:
            parts.append("<div class='alert'><b>⚠ News-driven divergences (V8b vs Chairman):</b><ul>")
            for div in divergences:
                parts.append(
                    f"<li><b>{div['ticker']}</b>: monthly quant says <b>{div['v8b']}</b>, "
                    f"news-aware chairman says <b>{div['chairman']}</b> "
                    f"(news tone {div.get('sentiment','neutral')}). "
                    f"Why: {div.get('rationale','')}</li>"
                )
            parts.append("</ul></div>")
        parts.append("<table>")
        parts.append("<tr><th>Ticker</th><th>Held?</th>"
                     "<th>5d action (W1)</th><th>1mo action (Chairman)</th>"
                     "<th>1mo conviction</th><th>News tone</th>"
                     "<th>Top deciding factor</th></tr>")
        for p in picks:
            tk = p["ticker"]
            v = verdict_map.get(tk, {})
            c = chair_per_pick.get(tk, {})
            is_held = tk.upper() in held
            held_cell = "Yes" if is_held else "No"

            # 5-day action: every W1 top pick is by definition a trading idea.
            # If held → "TRIM-or-HOLD" (we don't double up on a fresh weekly
            # signal for a name already in the book unless explicitly aligned).
            # Not held → "BUY (5d)" — the trade plan executes this.
            w1_action = "ADD (5d)" if is_held else "BUY (5d)"

            # 1-month action from chairman's verdict, position-aware
            chair_verdict = c.get("verdict")
            chair_action = position_aware_action(chair_verdict, is_held)

            # Color the chair pill by action class
            ca_cls = ("buy" if chair_action in ("BUY", "ADD")
                      else "accum" if chair_action == "WATCH"
                      else "hold" if chair_action == "HOLD"
                      else "reduce")
            w1_cls = "buy"

            decider = ""
            if c.get("deciding_factors"):
                decider = c["deciding_factors"][0][:80]
            parts.append(
                f"<tr><td><b>{tk}</b></td>"
                f"<td>{held_cell}</td>"
                f"<td><span class='pill {w1_cls}'>{w1_action}</span></td>"
                f"<td><span class='pill {ca_cls}'>{chair_action}</span> "
                f"<span style='color:#999;font-size:11px'>({chair_verdict or '—'})</span></td>"
                f"<td>{c.get('conviction','—')}</td>"
                f"<td>{c.get('sentiment_label','—')} ({c.get('sentiment_score','—')})</td>"
                f"<td style='font-size:12px;color:#444'>{decider}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            "<p style='color:#666;font-size:11px;margin-top:6px'>"
            "Action key — held: <b>ADD</b> top up · <b>HOLD</b> keep · "
            "<b>TRIM</b> lighten · <b>EXIT</b> close. "
            "Unheld: <b>BUY</b> open new · <b>WATCH</b> bench · <b>SKIP</b> don't enter. "
            "The trade plan below executes the 5-day BUY signal. "
            "If 5d says BUY but 1mo says SKIP/WATCH, treat the trade as short-horizon only — "
            "exit by the time stop, don't roll into a long-term hold."
            "</p>"
        )

    # PER-PICK TECHNICAL + FUNDAMENTAL + NEWS SUMMARY
    techs = b.get("technicals_per_pick") or {}
    funds = b.get("fundamentals_per_pick") or {}
    news_pp = b.get("news_per_pick") or {}
    if picks and (techs or funds):
        parts.append("<h2>Per-Pick Analyst Read</h2>")
        for p in picks:
            tk = p["ticker"]
            t = techs.get(tk) or {}
            f = funds.get(tk) or {}
            n = news_pp.get(tk) or {}
            parts.append(f"<div class='summary-box' style='margin-top:12px'>")
            parts.append(f"<h3 style='margin:0 0 6px;color:#0d3e66'>{tk} — {p.get('name','')}</h3>")

            # Technicals row
            parts.append("<p style='margin:4px 0'><b>Technicals:</b> ")
            if t.get("error"):
                parts.append(f"<span style='color:#666'>unavailable — {t['error']}</span>")
            else:
                parts.append(" · ".join(t.get("bullets") or []) or "no signals")
            parts.append("</p>")

            # Fundamentals row
            parts.append("<p style='margin:4px 0'><b>Fundamentals:</b> ")
            if f.get("error"):
                parts.append(f"<span style='color:#666'>unavailable — {f['error']}</span>")
            else:
                parts.append(" · ".join(f.get("bullets") or []) or "no fundamentals")
            parts.append("</p>")

            # News row (with sentiment)
            sent = (n.get("sentiment") or {})
            label = sent.get("label", "neutral")
            agg = sent.get("aggregate_score", 0) or 0
            label_cls = "pos" if agg > 0.1 else "neg" if agg < -0.1 else ""
            parts.append(f"<p style='margin:4px 0'><b>Recent news:</b> "
                         f"<span class='{label_cls}'>{label.replace('_',' ')}</span> "
                         f"(tone {agg:+.2f}, {sent.get('headline_count',0)} hdl)")
            heads = n.get("headlines") or []
            if heads:
                parts.append("<ul style='margin:4px 0 0;padding-left:18px;font-size:12px'>")
                for h in heads[:3]:
                    title = h.get("title", "").strip()
                    url = h.get("url")
                    src = h.get("source") or h.get("lang", "").upper()
                    date = h.get("date") or ""
                    if url:
                        parts.append(f"<li><a href='{url}' style='color:#0d3e66'>{title}</a> "
                                     f"<span style='color:#888'>— {src} {date}</span></li>")
                    else:
                        parts.append(f"<li>{title} <span style='color:#888'>— {src} {date}</span></li>")
                parts.append("</ul>")
            else:
                parts.append(" — no headlines</p>")
            parts.append("</p>")
            parts.append("</div>")

    # MARKET NEWS — portfolio-impact (or watchlist when flat)
    mn = b.get("market_news") or {}
    if mn and not mn.get("error"):
        basis = mn.get("filter_basis", "portfolio")
        target_set = mn.get("filter_target_tickers") or []
        if basis == "portfolio":
            heading = "Market News — Portfolio Impact"
            filter_explainer = (
                f"Filtered against your <b>{len(target_set)} held names</b> "
                f"({', '.join(target_set[:5])}{', …' if len(target_set) > 5 else ''}). "
                "Direct = headline names a holding. Sector = matches a held sector."
            )
        elif basis == "watchlist":
            heading = "Market News — Watchlist Impact (flat portfolio)"
            filter_explainer = (
                "No positions found in your portfolio CSV — filtering against "
                f"today's W1 candidates instead: <b>{', '.join(target_set)}</b>. "
                "Direct = headline names a candidate. Sector = matches a candidate sector."
            )
        else:
            heading = "Market News"
            filter_explainer = "No portfolio or watchlist available — showing all market headlines."

        parts.append(f"<h2>{heading}</h2>")
        msent = mn.get("sentiment") or {}
        agg = msent.get("aggregate_score", 0) or 0
        label = msent.get("label", "neutral")
        label_cls = "pos" if agg > 0.1 else "neg" if agg < -0.1 else ""
        raw_n = mn.get("raw_headline_count", 0)
        filt_n = mn.get("filtered_headline_count", len(mn.get("headlines") or []))
        parts.append("<div class='summary-box'>")
        parts.append(f"<p style='margin:0 0 4px'><b>Filtered:</b> kept {filt_n} of {raw_n} market headlines. "
                     f"<b>Tone:</b> <span class='{label_cls}'>{label.replace('_',' ')}</span> "
                     f"(score {agg:+.2f}).</p>")
        parts.append(f"<p style='margin:0;font-size:11px;color:#666'>{filter_explainer}</p>")
        parts.append("</div>")

        heads = mn.get("headlines") or []
        if heads:
            parts.append("<table>")
            parts.append("<tr><th>Impact</th><th>Source</th><th>Lang</th><th>Headline</th></tr>")
            for h in heads[:15]:
                title = h.get("title", "").strip()
                url = h.get("url")
                src = h.get("source") or "—"
                lang = h.get("lang", "").upper()
                impact = h.get("impact_level") or ""
                tickers = h.get("matched_tickers") or []
                sectors = h.get("matched_sectors") or []
                if impact == "direct":
                    badge = f"<span class='pill buy'>{', '.join(tickers)}</span>"
                elif impact == "sector":
                    badge = f"<span class='pill accum'>{', '.join(sectors)}</span>"
                else:
                    badge = f"<span class='pill hold'>{impact}</span>"
                title_cell = f"<a href='{url}' style='color:#0d3e66'>{title}</a>" if url else title
                parts.append(f"<tr><td>{badge}</td><td>{src}</td><td>{lang}</td><td>{title_cell}</td></tr>")
            parts.append("</table>")
        else:
            parts.append("<p style='color:#666;font-size:13px'>No market news touches your "
                         "current holdings today. (Macro headlines about CBE / EGP / rates are "
                         "filtered out by default — they affect everyone.)</p>")

    # TRADE PLANS — entry, stop, target, sizing per pick
    plans = b.get("trade_plans") or {}
    params = b.get("trade_plan_params") or {}
    if plans and any(not v.get("error") for v in plans.values()):
        parts.append("<h2>Trade Plans — Entry / Stop / Target</h2>")
        parts.append(
            f"<p style='color:#666;font-size:12px'>"
            f"Sized for NAV <b>{params.get('portfolio_nav_egp', 0):,.0f} EGP</b>, "
            f"risk <b>{params.get('risk_pct_per_trade')}%</b> per trade, "
            f"stop at <b>{ATR_STOP_MULTIPLE}× ATR</b>, "
            f"hold up to <b>{params.get('expected_hold_days')} sessions</b>. "
            "Override via env vars: EGX_PORTFOLIO_NAV_EGP, EGX_RISK_PCT_PER_TRADE, EGX_ATR_STOP_MULTIPLE."
            "</p>"
        )
        parts.append("<table>")
        parts.append("<tr><th>Ticker</th><th>Action</th><th>Entry limit</th>"
                     "<th>Stop-loss</th><th>Scale-out (1R)</th><th>Target (1.5R)</th>"
                     "<th>Shares</th><th>Position</th><th>Risk EGP</th><th>Time stop</th></tr>")
        for tk, plan in plans.items():
            if plan.get("error"):
                parts.append(f"<tr><td><b>{tk}</b></td><td colspan='9'>"
                             f"Trade plan error: {plan['error']}</td></tr>")
                continue
            stop_dist = plan.get("stop_distance_egp")
            entry = plan.get("entry_limit_egp")
            stop = plan.get("stop_loss_egp")
            scale_out = plan.get("scale_out_egp")
            target = plan.get("target_egp")
            scale_sh = plan.get("scale_out_shares") or 0
            run_sh = plan.get("runner_shares") or 0
            stop_pct = (stop_dist / entry * 100) if (stop_dist and entry) else 0
            scale_pct = ((scale_out / entry - 1) * 100) if (scale_out and entry) else 0
            target_pct = ((target / entry - 1) * 100) if (target and entry) else 0
            parts.append(
                f"<tr>"
                f"<td><b>{tk}</b></td>"
                f"<td><span class='pill buy'>{plan.get('action')}</span></td>"
                f"<td><b>{entry}</b><br><span style='color:#666;font-size:11px'>last {plan.get('last_close_egp')}</span></td>"
                f"<td class='neg'><b>{stop}</b><br><span style='font-size:11px'>−{stop_pct:.1f}%</span></td>"
                f"<td class='pos'><b>{scale_out}</b><br><span style='font-size:11px'>sell {scale_sh:,} sh (+{scale_pct:.1f}%)</span></td>"
                f"<td class='pos'><b>{target}</b><br><span style='font-size:11px'>{run_sh:,} sh OR trail (+{target_pct:.1f}%)</span></td>"
                f"<td>{plan.get('shares'):,}</td>"
                f"<td>{plan.get('position_cost_egp'):,.0f} EGP<br><span style='color:#666;font-size:11px'>{plan.get('position_pct_of_nav')}% NAV</span></td>"
                f"<td>{plan.get('risk_egp'):,.0f}</td>"
                f"<td>{plan.get('time_stop_date')}</td>"
                f"</tr>"
            )
        parts.append("</table>")

        # Add a clean "When to sell" recap card per pick
        parts.append("<h3 style='font-size:14px;margin-top:14px'>Exit rules (apply to every pick)</h3>")
        parts.append("<div class='summary-box'><ol style='margin:0;padding-left:20px'>")
        parts.append("<li><b>Scale-out at 1R</b> — when price reaches the scale-out level, sell half and move the stop on the remainder to breakeven. Locks in the win and removes downside on the runner.</li>")
        parts.append("<li><b>Target or trail (1.5R)</b> — sell the runner at the target, OR trail by 1×ATR on new highs to let it run. Pick whichever you'll actually execute.</li>")
        parts.append("<li><b>Stop loss</b> — exit immediately if the stop level trades. Use a stop-loss order with the broker, not a mental stop.</li>")
        parts.append("<li><b>Time stop (conditional)</b> — applies ONLY if scale-out never triggered. Once half is off at 1R, the runner trails freely past the time stop.</li>")
        parts.append("<li><b>Catalyst override</b> — if a blocking catalyst (e.g., earnings) lands during the hold, exit before the print.</li>")
        parts.append("<li><b>Re-entry</b> — if the same ticker reappears in next Sunday's W1 top-5, extend the trail rather than close-and-reopen.</li>")
        parts.append("</ol></div>")

    # PORTFOLIO
    parts.append("<h2>Your Portfolio</h2>")
    if isinstance(pf, dict) and "error" in pf:
        parts.append(f"<p>Portfolio not loaded: {pf['error']}</p>")
        parts.append(f"<p>Hint: {pf.get('hint', 'Set EGX_PORTFOLIO_CSV env var.')}</p>")
    elif pf_positions:
        total_pnl_pct = pf.get("total_pnl_pct", 0)
        total_pnl_egp = pf.get("total_pnl_egp", 0)
        cls = "pos" if total_pnl_pct >= 0 else "neg"
        parts.append("<div class='summary-box'>")
        parts.append(f"<p>Total cost: <b>{pf.get('total_cost_egp', 0):,.0f} EGP</b> | "
                     f"Market value: <b>{pf.get('total_value_egp', 0):,.0f} EGP</b> | "
                     f"P&L: <b class='{cls}'>{total_pnl_egp:+,.0f} EGP ({total_pnl_pct:+.2f}%)</b></p>")
        parts.append("</div>")

        # Portfolio vs market
        market_ytd = b.get("market_ytd_pct")
        if market_ytd is not None:
            spread = total_pnl_pct - market_ytd
            cls = "pos" if spread > 0 else "neg"
            parts.append(f"<p><b>vs market:</b> portfolio {total_pnl_pct:+.2f}% / "
                         f"market YTD {market_ytd:+.2f}% / "
                         f"<span class='{cls}'>active {spread:+.2f}pp</span></p>")

        parts.append("<table>")
        parts.append("<tr><th>Ticker</th><th>Shares</th><th>Cost</th><th>Live</th>"
                     "<th>Mkt Value</th><th>P&L</th><th>P&L %</th><th>Wgt</th></tr>")
        for pos in pf_positions:
            pnl_pct = pos.get("unrealized_pnl_pct", 0) or 0
            cls = "pos" if pnl_pct > 0 else "neg"
            parts.append(
                f"<tr><td><b>{pos['ticker']}</b></td>"
                f"<td>{pos['shares']:,.0f}</td>"
                f"<td>{pos['cost_basis']:,.2f}</td>"
                f"<td>{pos.get('current_price', '—')}</td>"
                f"<td>{pos.get('market_value', '—'):,.0f}</td>"
                f"<td class='{cls}'>{pos.get('unrealized_pnl', 0):+,.0f}</td>"
                f"<td class='{cls}'>{pnl_pct:+.2f}%</td>"
                f"<td>{pos.get('weight_pct', 0)}%</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p>No positions in portfolio CSV.</p>")

    # FOOTER
    parts.append("<div class='footer'>")
    parts.append(f"<p>Generated by EGX MCP daily briefing routine. "
                 f"Built on V8b production (monthly) and W1 weekly models — both validated against 28 months of EGX data.</p>")
    parts.append(f"<p>Validation: V8b Sharpe 1.27 / W1 Sharpe 1.64 (full window) → V8b 3.29 / W1 2.78 (holdout OOS). "
                 f"Both beat the broad EGX market by +29 to +67pp annualized.</p>")
    parts.append("<p><b>Disclaimer:</b> Algorithmic verdicts for research only. Not investment advice. "
                 "Verify quotes against EGX official tape before trading. EGX trading session: 10:00 to 14:30 Cairo.</p>")
    parts.append("</div></body></html>")

    return "".join(parts)


def render_text(b: dict) -> str:
    """Plain-text version for any client that doesn't render HTML."""
    if b.get("skip"):
        return f"Skipped: {b.get('reason')}"
    lines = []
    lines.append(f"=== EGX PRE-MARKET BRIEFING — {b['cairo_local']} ===\n")

    macro = b.get("macro", {}) or {}
    regime_d = b.get("regime", {}) or {}
    egp = (macro.get("egp_usd") or {}).get("value")
    egp_chg = (macro.get("egp_usd") or {}).get("change_pct")
    brent = (macro.get("brent_usd") or {}).get("value")
    brent_chg = (macro.get("brent_usd") or {}).get("change_pct")
    lines.append("--- MACRO ---")
    egx30 = b.get("egx30_spot") or {}
    egx70 = b.get("egx70_spot") or {}
    if not egx30.get("error") and egx30.get("value") is not None:
        ch = egx30.get("change_pct")
        chg = f"{ch:+.2f}%" if ch is not None else "—"
        lines.append(f"  EGX 30:  {egx30['value']:,.0f} pts ({chg})")
    if not egx70.get("error") and egx70.get("value") is not None:
        ch = egx70.get("change_pct")
        chg = f"{ch:+.2f}%" if ch is not None else "—"
        lines.append(f"  EGX 70:  {egx70['value']:,.0f} pts ({chg})")
    if egp is not None:
        chg = f"{egp_chg:+.2f}%" if egp_chg is not None else "—"
        lines.append(f"  USD/EGP: {egp} ({chg})")
    if brent is not None:
        chg = f"{brent_chg:+.2f}%" if brent_chg is not None else "—"
        lines.append(f"  Brent:   ${brent:.2f} ({chg})")
    gold = b.get("gold_egp") or {}
    if not gold.get("error") and gold.get("egp_per_gram_24k") is not None:
        lines.append(f"  Gold 24K: {gold['egp_per_gram_24k']:,.2f} EGP/g  |  "
                     f"21K: {gold['egp_per_gram_21k']:,.2f}  |  "
                     f"18K: {gold['egp_per_gram_18k']:,.2f}")
        lines.append(f"  Egyptian gold pound: {gold['egyptian_gold_pound_egp']:,.0f} EGP "
                     "(8 g of 21K — local dealer ≈ +5-15%)")
    lines.append(f"  Regime:  {regime_d.get('regime', '?')} — {regime_d.get('description','')[:80]}")
    lines.append("")

    w1 = b.get("w1_picks", {}) or {}
    picks = w1.get("top_picks", []) or []
    lines.append(f"--- TODAY'S W1 PICKS (5-day horizon, eligible: {w1.get('n_eligible','?')}) ---")
    for i, p in enumerate(picks, 1):
        tk = p['ticker']
        chart = tv_scraper.chart_url(tk)
        rsi_v = p.get('rsi_14')
        rsi_str = f" RSI={rsi_v:.0f}" if rsi_v is not None else ""
        lines.append(f"  {i}. {tk:6s} score={p.get('score','?')}  "
                     f"price={p.get('price','—')}  "
                     f"5d={p.get('mom_5d_pct',0):+.1f}%{rsi_str}  "
                     f"trend={'^' if p.get('above_ma20') else 'v'}")
        lines.append(f"     chart: {chart}")
    exdiv_filtered = b.get("exdiv_filtered") or []
    if exdiv_filtered:
        lines.append("  Dropped (ex-dividend within 3 sessions):")
        for d in exdiv_filtered:
            lines.append(f"    {d['ticker']} (ex-div {d.get('ex_dividend_date')}, {d['days_to_exdividend']}d out)")

    plans = b.get("trade_plans") or {}
    if plans:
        lines.append("\n--- TRADE PLANS ---")
        params = b.get("trade_plan_params") or {}
        lines.append(f"  NAV {params.get('portfolio_nav_egp', 0):,.0f} EGP | "
                     f"risk {params.get('risk_pct_per_trade')}%/trade | "
                     f"hold up to {params.get('expected_hold_days')}d")
        for tk, plan in plans.items():
            if plan.get("error"):
                lines.append(f"  {tk}: trade plan error - {plan['error'][:60]}")
                continue
            lines.append(f"  {tk}: ENTRY {plan.get('entry_limit_egp')} | "
                         f"STOP {plan.get('stop_loss_egp')} | "
                         f"SCALE-OUT {plan.get('scale_out_egp')} ({plan.get('scale_out_shares')} sh) | "
                         f"TARGET {plan.get('target_egp')} ({plan.get('runner_shares')} sh) | "
                         f"{plan.get('shares')} total ({plan.get('position_pct_of_nav')}% NAV) | "
                         f"R:R 1:{plan.get('reward_to_risk')} | "
                         f"time stop {plan.get('time_stop_date')}")
        lines.append("  EXIT RULES: at scale-out → sell half + stop to BE | at target → sell rest OR trail by 1×ATR on new highs")
        lines.append("              stop hit → exit | time stop applies only if scale-out never triggered")

    # Action recommendations — position-aware, two horizons
    chair_per_pick = b.get("chairman_per_pick") or {}
    divergences = b.get("divergences") or []
    held = held_set_from_portfolio(b)
    if picks and chair_per_pick:
        lines.append("\n--- ACTION RECOMMENDATIONS ---")
        lines.append("  5-day = W1 weekly trading rank. 1-month = Chairman synthesis (news-aware).")
        lines.append("  Actions are position-aware: held -> ADD/HOLD/TRIM/EXIT, unheld -> BUY/WATCH/SKIP.")
        lines.append("")
        for p in picks:
            tk = p["ticker"]
            c = chair_per_pick.get(tk, {})
            is_held = tk.upper() in held
            held_cell = "HELD " if is_held else "FRESH"
            w1_action = "ADD (5d)" if is_held else "BUY (5d)"
            chair_verdict = c.get("verdict")
            chair_action = position_aware_action(chair_verdict, is_held)
            tone = c.get("sentiment_label", "—")
            lines.append(f"  {tk:6s} {held_cell}  5d: {w1_action:<10s}  "
                         f"1mo: {chair_action:<6s} ({chair_verdict or '—'})  tone: {tone}")
        if divergences:
            lines.append("\n  News-driven divergences (V8b vs Chairman, monthly horizon) — INFORMATIONAL:")
            lines.append("  Tracked for forward grading. Does NOT gate W1 trade plans.")
            for div in divergences:
                lines.append(f"    {div['ticker']}: V8b says {div['v8b']} but chairman "
                             f"says {div['chairman']} — {div.get('rationale','')[:100]}")
        lines.append("")
        lines.append("  Key: ADD/HOLD/TRIM/EXIT for held names, BUY/WATCH/SKIP for unheld.")
        lines.append("  Trade plan executes the 5-day BUY. If 1-month says SKIP/WATCH,")
        lines.append("  treat as short-horizon only — exit by the time stop, don't roll into a long hold.")

    # Per-pick analyst read
    techs = b.get("technicals_per_pick") or {}
    funds = b.get("fundamentals_per_pick") or {}
    news_pp = b.get("news_per_pick") or {}
    if picks and (techs or funds):
        lines.append("\n--- PER-PICK ANALYST READ ---")
        for p in picks:
            tk = p["ticker"]
            t = techs.get(tk) or {}
            f = funds.get(tk) or {}
            n = news_pp.get(tk) or {}
            lines.append(f"  {tk}:")
            t_line = "; ".join(t.get("bullets") or []) if not t.get("error") else f"unavailable ({t.get('error','')})"
            f_line = "; ".join(f.get("bullets") or []) if not f.get("error") else f"unavailable ({f.get('error','')})"
            lines.append(f"    Technicals:    {t_line}")
            lines.append(f"    Fundamentals:  {f_line}")
            sent = (n.get("sentiment") or {})
            lines.append(f"    News tone:     {sent.get('label','neutral')} "
                         f"({sent.get('aggregate_score', 0):+.2f}, "
                         f"{sent.get('headline_count', 0)} hdl)")
            for h in (n.get("headlines") or [])[:2]:
                title = (h.get("title") or "").strip()[:90]
                src = h.get("source") or h.get("lang", "").upper()
                lines.append(f"      - {title} [{src}]")

    # Market news — portfolio-impact (or watchlist when flat)
    mn = b.get("market_news") or {}
    if mn and not mn.get("error"):
        msent = mn.get("sentiment") or {}
        raw_n = mn.get("raw_headline_count", 0)
        filt_n = mn.get("filtered_headline_count", len(mn.get("headlines") or []))
        basis = mn.get("filter_basis", "portfolio")
        target_set = mn.get("filter_target_tickers") or []
        if basis == "portfolio":
            lines.append("\n--- MARKET NEWS — PORTFOLIO IMPACT ---")
            lines.append(f"  Filtered against {len(target_set)} held names.")
        elif basis == "watchlist":
            lines.append("\n--- MARKET NEWS — WATCHLIST IMPACT (flat portfolio) ---")
            lines.append(f"  No positions; filtering against W1 candidates: {', '.join(target_set)}")
        else:
            lines.append("\n--- MARKET NEWS ---")
            lines.append("  (no portfolio or watchlist — showing all)")
        lines.append(f"  Filtered: kept {filt_n} of {raw_n} headlines.")
        lines.append(f"  Tone: {msent.get('label','neutral')} "
                     f"(score {msent.get('aggregate_score', 0):+.2f})")
        heads = mn.get("headlines") or []
        if not heads:
            lines.append("  (no relevant headlines today)")
        for h in heads[:12]:
            title = (h.get("title") or "").strip()[:100]
            src = h.get("source") or h.get("lang", "").upper()
            tickers = h.get("matched_tickers") or []
            sectors = h.get("matched_sectors") or []
            if tickers:
                tag = "[" + ",".join(tickers) + "]"
            elif sectors:
                tag = "[sector:" + ",".join(sectors) + "]"
            else:
                tag = "[" + (h.get("impact_level") or "?") + "]"
            lines.append(f"  {tag} {title} — {src}")

    pf = b.get("portfolio", {}) or {}
    if isinstance(pf, dict) and pf.get("positions"):
        lines.append("\n--- PORTFOLIO ---")
        lines.append(f"  Cost: {pf.get('total_cost_egp', 0):,.0f} | "
                     f"Value: {pf.get('total_value_egp', 0):,.0f} | "
                     f"P&L: {pf.get('total_pnl_egp', 0):+,.0f} ({pf.get('total_pnl_pct', 0):+.2f}%)")
        market_ytd = b.get("market_ytd_pct")
        if market_ytd is not None:
            spread = pf.get('total_pnl_pct', 0) - market_ytd
            lines.append(f"  vs Market YTD {market_ytd:+.2f}%: active {spread:+.2f}pp")

    lines.append("\nDisclaimer: Research only, not investment advice.")
    return "\n".join(lines)


def send_email(html: str, text: str, subject: str) -> bool:
    """Send via SMTP if all env vars are set. Returns True if sent."""
    host = os.environ.get("SMTP_HOST")
    port = os.environ.get("SMTP_PORT", "587")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    to_addr = os.environ.get("BRIEFING_EMAIL_TO")
    if not all([host, user, password, to_addr]):
        print("SMTP env vars not set — skipping email send.")
        print("  Required: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, BRIEFING_EMAIL_TO")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(host, int(port), timeout=30) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        print(f"Email sent to {to_addr}")
        return True
    except Exception as e:
        print(f"Email send failed: {e}")
        return False


def main():
    force = "--force" in sys.argv
    print(f"=== Daily briefing run @ {datetime.utcnow().isoformat()} UTC ===")

    b = build_briefing(force=force)
    if b.get("skip"):
        print(f"Skipped: {b['reason']}")
        return 0

    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    html_path = OUTPUT_DIR / f"briefing_{today_str}.html"
    txt_path = OUTPUT_DIR / f"briefing_{today_str}.txt"
    json_path = OUTPUT_DIR / f"briefing_{today_str}.json"

    html = render_html(b)
    text = render_text(b)

    html_path.write_text(html, encoding="utf-8")
    txt_path.write_text(text, encoding="utf-8")
    json_path.write_text(json.dumps(b, indent=2, default=str), encoding="utf-8")

    print(f"Briefing written:")
    print(f"  {html_path}")
    print(f"  {txt_path}")
    print(f"  {json_path}")

    # Print key stats to console for Task Scheduler logs
    w1 = b.get("w1_picks", {}) or {}
    picks = w1.get("top_picks", []) or []
    print(f"\nTop picks today:")
    for i, p in enumerate(picks, 1):
        print(f"  {i}. {p['ticker']:6s} score={p.get('score')}")
    pf = b.get("portfolio", {}) or {}
    if isinstance(pf, dict) and pf.get("total_pnl_pct") is not None:
        market_ytd = b.get("market_ytd_pct")
        print(f"\nPortfolio P&L: {pf.get('total_pnl_pct'):+.2f}%")
        if market_ytd is not None:
            spread = pf['total_pnl_pct'] - market_ytd
            print(f"vs Market YTD {market_ytd:+.2f}%: {spread:+.2f}pp")

    # Try to email
    subject = f"EGX Briefing — {today_str} ({len(picks)} picks)"
    sent = send_email(html, text, subject)
    if not sent:
        print(f"\nTo enable email, set these env vars and re-run:")
        print(f"  SMTP_HOST=smtp.gmail.com")
        print(f"  SMTP_PORT=587")
        print(f"  SMTP_USER=your.email@gmail.com")
        print(f"  SMTP_PASS=<gmail-app-password>")
        print(f"  BRIEFING_EMAIL_TO=where.to.send@example.com")

    return 0


if __name__ == "__main__":
    sys.exit(main())
