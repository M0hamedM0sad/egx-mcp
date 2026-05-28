"""Per-company comprehensive brief — every signal we have on one name.

Aggregates the entire MCP's intelligence about a single ticker into a
structured snapshot a PM can read in 30 seconds. Pulls from:

    quote          live price / volume / change
    fundamentals   sanitized P/E, P/B, ROE, EPS, BVPS (Mubasher + Yahoo)
    technicals     RSI, MACD, MA cross, Bollinger, ATR
    weekly_score   W1 5-day model rank + drivers
    decision       monthly V8b verdict (BUY/HOLD/SELL)
    peers          sector ranking
    factors        EGX/EGP/Brent/Gold betas
    macro          regime + sector bias
    calendar       earnings date, ex-div, recent disclosures
    news           latest 5 headlines
    short_term     bootstrap MC 5-day forecast (E[ret], P(up>2%), 90% CI)

Everything is wrapped in error handlers so a partial failure doesn't kill
the whole brief — missing fields just appear as null.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from . import (
    market, fundamentals, technicals, peers, factors, macro,
    decision, simulation, news, weekly,
)
from . import calendar as cal_mod
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.company_brief")


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"_error": str(e), "_function": fn.__name__}


def brief(ticker: str, portfolio_value_egp: float | None = None) -> dict[str, Any]:
    """Return the comprehensive intelligence brief for one EGX name."""
    canonical, yahoo, name = resolve_ticker(ticker)

    quote = _safe(market.get_quote, canonical)
    fund = _safe(fundamentals.get_fundamentals, canonical)
    tech = _safe(technicals.compute, canonical)
    dec = _safe(decision.decide, canonical, portfolio_value_egp=portfolio_value_egp)
    peer = _safe(peers.compare, canonical, max_peers=8)
    fac = _safe(factors.ticker_factor_exposure, canonical)
    cal = _safe(cal_mod.get_calendar, canonical)
    en_news = _safe(news.fetch, canonical, "en", 5)
    sim = _safe(simulation.simulate_one, canonical, 5, 1500, 60)

    # Get weekly model rank for this ticker
    try:
        wk_full = weekly.rank_universe()
        wk_ranked = wk_full.get("top_picks", []) + wk_full.get("runners_up", [])
        # Look in all_ranked too — rank_universe only returns top + runners up,
        # so search the eligible set if not already there
        all_eligible = wk_full.get("top_picks", []) + wk_full.get("runners_up", [])
        wk_row = next((r for r in all_eligible if r.get("ticker") == canonical), None)
        wk_rank = next((i + 1 for i, r in enumerate(wk_ranked)
                        if r.get("ticker") == canonical), None)
        wk_summary = {
            "rank_in_top_10": wk_rank,
            "score": wk_row.get("score") if wk_row else None,
            "in_top_picks": (wk_rank is not None and wk_rank <= 5),
        }
    except Exception as e:
        wk_summary = {"_error": str(e)}

    # Macro fit for this stock's sector
    sector = (fund or {}).get("sector") if isinstance(fund, dict) else None
    if sector and sector not in ("Unknown", None):
        macro_bias = _safe(macro.sector_macro_bias, sector)
    else:
        macro_bias = None

    # Pull Mubasher fundamentals snapshot (the audited values)
    audited = None
    try:
        from . import mubasher_fundamentals
        audited = mubasher_fundamentals.get_fundamentals(canonical)
    except Exception:
        pass

    # Build the structured brief
    return {
        "ticker": canonical,
        "yahoo_symbol": yahoo,
        "name": (quote or {}).get("name") or name,
        "as_of": datetime.utcnow().isoformat() + "Z",

        "snapshot": {
            "price": (quote or {}).get("price"),
            "change_pct_today": (quote or {}).get("change_pct"),
            "day_high": (quote or {}).get("day_high"),
            "day_low": (quote or {}).get("day_low"),
            "volume_today": (quote or {}).get("volume"),
            "avg_volume_20d": (quote or {}).get("avg_volume"),
            "market_cap_egp": (quote or {}).get("market_cap"),
            "52w_high": (quote or {}).get("52w_high"),
            "52w_low": (quote or {}).get("52w_low"),
        },

        "fundamentals": {
            "sector": sector,
            "pe_ratio": (fund or {}).get("pe_ratio"),
            "pb_ratio": (fund or {}).get("pb_ratio"),
            "roe_pct": (fund or {}).get("roe_pct"),
            "trailing_eps": (fund or {}).get("trailing_eps"),
            "book_value_per_share": (fund or {}).get("book_value_per_share"),
            "dividend_yield_pct": (fund or {}).get("dividend_yield_pct"),
            "fundamentals_source": (fund or {}).get("fundamentals_source"),
            "passes_quality_filter_v8b": (
                (fund or {}).get("roe_pct") is not None
                and (fund or {}).get("roe_pct") >= 10
            ),
            "audited_from_mubasher": (
                audited.get("source") if audited and not audited.get("error") else None
            ),
        },

        "technicals": {
            "as_of": (tech or {}).get("as_of"),
            "rsi_14": ((tech or {}).get("indicators") or {}).get("rsi_14"),
            "macd": ((tech or {}).get("indicators") or {}).get("macd"),
            "macd_signal": ((tech or {}).get("indicators") or {}).get("macd_signal"),
            "sma_20": ((tech or {}).get("indicators") or {}).get("sma_20"),
            "sma_50": ((tech or {}).get("indicators") or {}).get("sma_50"),
            "sma_200": ((tech or {}).get("indicators") or {}).get("sma_200"),
            "bb_upper": ((tech or {}).get("indicators") or {}).get("bb_upper"),
            "bb_lower": ((tech or {}).get("indicators") or {}).get("bb_lower"),
            "atr_14": ((tech or {}).get("indicators") or {}).get("atr_14"),
            "signals": (tech or {}).get("signals"),
        },

        "weekly_model_w1": wk_summary,

        "monthly_decision_v8b": {
            "verdict": (dec or {}).get("verdict"),
            "conviction": (dec or {}).get("conviction"),
            "composite_score": (dec or {}).get("composite_score"),
            "fair_value": (dec or {}).get("fair_value_estimate"),
            "upside_pct": (dec or {}).get("upside_pct"),
            "key_drivers": ((dec or {}).get("key_drivers") or [])[:5],
            "key_risks": ((dec or {}).get("key_risks") or [])[:5],
            "blocking_catalysts": (dec or {}).get("blocking_catalysts") or [],
        },

        "short_term_simulation_5d": {
            "expected_return_pct": (sim or {}).get("expected_return_pct"),
            "prob_up_2pct": (sim or {}).get("prob_up_2pct"),
            "prob_up_5pct": (sim or {}).get("prob_up_5pct"),
            "prob_down_5pct": (sim or {}).get("prob_down_5pct"),
            "p10_return_pct": (sim or {}).get("p10_return_pct"),
            "p90_return_pct": (sim or {}).get("p90_return_pct"),
            "imminent_move_score": (sim or {}).get("imminent_move_score"),
            "edge_drivers": ((sim or {}).get("edge_drivers") or [])[:3],
        },

        "peer_context": {
            "sector": (peer or {}).get("sector"),
            "rank_in_sector": (peer or {}).get("target_rank_in_sector"),
            "relative": (peer or {}).get("target_relative_to_peers"),
            "peer_count": (peer or {}).get("peer_count"),
        },

        "factor_exposures_90d": {
            "egx30_beta": ((fac or {}).get("factor_betas") or {}).get("egx30"),
            "egp_usd_beta": ((fac or {}).get("factor_betas") or {}).get("egp"),
            "brent_beta": ((fac or {}).get("factor_betas") or {}).get("brent"),
            "gold_beta": ((fac or {}).get("factor_betas") or {}).get("gold"),
            "em_beta": ((fac or {}).get("factor_betas") or {}).get("em"),
            "alpha_daily_pct": (fac or {}).get("alpha_daily_pct"),
            "r_squared": (fac or {}).get("r_squared"),
            "interpretation": (fac or {}).get("interpretation"),
        },

        "macro_fit": {
            "sector_bias": (macro_bias or {}).get("macro_bias") if macro_bias else None,
            "reasons": (macro_bias or {}).get("reasons") if macro_bias else [],
        },

        "catalysts": {
            "next_earnings_date": (cal or {}).get("next_earnings_date"),
            "days_to_earnings": (cal or {}).get("days_to_earnings"),
            "ex_dividend_date": (cal or {}).get("ex_dividend_date"),
            "recent_disclosures": ((cal or {}).get("recent_disclosures") or [])[:3],
            "blocking": (cal or {}).get("blocking", False),
            "flags": (cal or {}).get("catalyst_flags") or [],
        },

        "recent_news": [
            {"date": a.get("date"), "title": a.get("title"),
             "source": a.get("source"), "url": a.get("url")}
            for a in (en_news or {}).get("articles", [])[:5]
        ],

        "summary_for_pm": _build_summary(canonical, dec, wk_summary, fund,
                                          peer, cal, macro_bias, sim),

        "disclaimer": (
            "Comprehensive snapshot for research only. Not investment advice. "
            "Cross-check critical numbers (P/E, ROE, last close) against the "
            "EGX official tape before acting."
        ),
    }


def _build_summary(ticker, dec, wk, fund, peer, cal, macro_bias, sim) -> list[str]:
    """Generate a 5-7 line plain-English summary for fast scanning."""
    lines = []
    if dec:
        verdict = dec.get("verdict")
        conv = dec.get("conviction")
        if verdict:
            lines.append(f"Monthly model (V8b): {verdict} — {conv} conviction "
                         f"(score {dec.get('composite_score', '?')}/100)")
    if wk and wk.get("score") is not None:
        rank = wk.get("rank_in_top_10")
        if rank is not None and rank <= 5:
            lines.append(f"Weekly model (W1): TOP-{rank} pick this week (score {wk['score']})")
        elif rank is not None:
            lines.append(f"Weekly model (W1): runner-up rank {rank}")
        else:
            lines.append(f"Weekly model (W1): not in top picks (score {wk['score']})")
    if fund:
        roe = fund.get("roe_pct")
        pe = fund.get("pe_ratio")
        if roe is not None and pe is not None:
            quality_call = "high quality" if roe >= 15 else ("decent quality" if roe >= 10 else "low quality")
            value_call = "cheap" if pe < 8 else ("fair" if pe < 15 else "expensive")
            lines.append(f"Fundamentals: ROE {roe:.1f}% ({quality_call}), P/E {pe:.1f} ({value_call})")
    if peer:
        rel = peer.get("target_relative_to_peers")
        rank_p = peer.get("target_rank_in_sector")
        if rel and rank_p:
            lines.append(f"Peers: {rel.replace('_', ' ')} (rank {rank_p}/{peer.get('peer_count')})")
    if macro_bias and macro_bias.get("macro_bias"):
        bias = macro_bias["macro_bias"]
        direction = "tailwind" if bias > 0 else "headwind"
        lines.append(f"Macro: {direction} ({bias:+.2f}) — {'; '.join(macro_bias.get('reasons', [])[:2])}")
    if cal:
        d2e = cal.get("days_to_earnings")
        if d2e is not None and 0 <= d2e <= 14:
            lines.append(f"⚠ Earnings in {d2e} days — defer or size down")
    if sim:
        e_ret = sim.get("expected_return_pct")
        p_up = sim.get("prob_up_2pct")
        if e_ret is not None and p_up is not None:
            lines.append(f"5-day MC: E[ret] {e_ret:+.2f}%, P(up>2%) {p_up:.0%}")
    return lines
