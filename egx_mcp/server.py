"""
EGX MCP Server
==============

A Model Context Protocol server exposing live Egyptian Exchange (EGX) data
plus a local portfolio tracker fed by CSV.

Tools:
  Market data:
    - get_quote(ticker)              Live quote for a single security
    - get_history(ticker, period)    OHLCV history (1d-10y)
    - get_index(name)                EGX30 / EGX70 / EGX100 / EGX30 TR

  Discovery:
    - list_egx_stocks(sector?)       Browse the listed universe
    - screen_stocks(...)             Filter by PE, volume, market cap

  Intelligence:
    - get_disclosures(ticker, days)  Latest disclosures from egx.com.eg
    - get_news(ticker, lang)         News from Mubasher / Investing

  Analytics:
    - compute_technicals(ticker)     RSI, MACD, MAs, ATR, Bollinger
    - portfolio_summary()            Read CSV, compute live P&L

  Decision layer (the new headline tools):
    - get_fundamentals(ticker)       Sanitized P/E, P/B, ROE, margin, leverage
    - get_macro_context()            EGP/USD, Brent, CBE rates, regime flags
    - score_stock(ticker)            Composite 0-100 score (val/qual/mom/risk)
    - compare_peers(ticker)          Sector ranking with relative-value verdict
    - position_size(ticker, ...)     ATR-based shares, stop, target
    - get_catalyst_calendar(ticker)  Earnings, ex-div, disclosure clusters
    - decide(ticker, ...)            BUY/HOLD/SELL synthesizer with full rationale

Author: Mohamed Mosad (linkedin.com/in/m0hamedm0sad)
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# CRITICAL: All logs must go to stderr. stdout is reserved for JSON-RPC traffic.
# Anything written to stdout will corrupt the protocol stream and Claude Desktop
# will silently disconnect.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("egx-mcp")

from mcp.server.fastmcp import FastMCP

from .data import (
    market, disclosures, news, technicals, portfolio, universe,
    fundamentals, macro, scoring, peers, sizing, decision, simulation,
    risk_free, liquidity, regime, factors, risk as risk_mod,
    optimizer, backtest, weekly, company_brief,
    sentiment, debate as debate_mod, risk_gate as risk_gate_mod,
    reflection, agentic_backtest, behavior, price_cache, briefing_page,
    project_impact as project_impact_mod,
    events as events_mod, forecast as forecast_mod,
    ir_fetch as ir_fetch_mod, ir_extract as ir_extract_mod,
)
from .data import calendar as cal_mod

mcp = FastMCP("egx-mcp")


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------

@mcp.tool()
def get_quote(ticker: str) -> dict[str, Any]:
    """Fetch the latest live quote for an EGX-listed security.

    Use this for current price, day change, volume, and basic stats.
    Accepts either the EGX ISIN-style code (e.g. 'EGS65541C012') or the
    Yahoo-suffixed form ('EGS65541C012.CA'). Common nicknames like 'CIRA',
    'COMI', 'HRHO', 'SWDY' are also accepted.

    Args:
        ticker: EGX security code or common ticker.

    Returns:
        Dict with: ticker, name, price, change, change_pct, day_high,
        day_low, volume, market_cap, currency, timestamp.
    """
    return market.get_quote(ticker)


@mcp.tool()
def get_history(
    ticker: str,
    period: str = "6mo",
    interval: str = "1d",
) -> dict[str, Any]:
    """Fetch OHLCV history for an EGX security.

    Args:
        ticker: EGX code or nickname.
        period: One of '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'.
        interval: One of '1d', '1wk', '1mo'. Intraday not reliably available for EGX.

    Returns:
        Dict with: ticker, period, interval, rows (list of {date, open, high,
        low, close, volume}), summary (start_price, end_price, return_pct,
        max_drawdown_pct, volatility_pct).
    """
    return market.get_history(ticker, period=period, interval=interval)


@mcp.tool()
def get_index(name: str = "EGX30") -> dict[str, Any]:
    """Fetch the current value of an EGX index.

    Args:
        name: One of 'EGX30', 'EGX30TR' (total return), 'EGX70', 'EGX100',
              'EGX70EWI' (equal-weighted). Defaults to EGX30.

    Returns:
        Dict with: index, value, change, change_pct, day_high, day_low,
        ytd_return_pct, timestamp.
    """
    return market.get_index(name)


# ---------------------------------------------------------------------------
# DISCOVERY
# ---------------------------------------------------------------------------

@mcp.tool()
def list_egx_stocks(sector: str | None = None) -> dict[str, Any]:
    """List EGX-listed equities, optionally filtered by sector.

    Args:
        sector: Optional sector filter. Examples: 'Banks', 'Real Estate',
                'Industrial Goods', 'Food', 'Telecom', 'Healthcare'.

    Returns:
        Dict with: count, sector, stocks (list of {ticker, name, sector,
        market_cap, last_price}).
    """
    return universe.list_stocks(sector=sector)


@mcp.tool()
def screen_stocks(
    min_pe: float | None = None,
    max_pe: float | None = None,
    min_market_cap_egp_m: float | None = None,
    min_avg_volume: int | None = None,
    sector: str | None = None,
    sort_by: str = "market_cap",
) -> dict[str, Any]:
    """Screen EGX stocks by fundamental and liquidity filters.

    Args:
        min_pe: Minimum P/E ratio.
        max_pe: Maximum P/E ratio.
        min_market_cap_egp_m: Minimum market cap in EGP millions.
        min_avg_volume: Minimum average daily volume (shares).
        sector: Sector filter (see list_egx_stocks).
        sort_by: 'market_cap', 'pe', 'volume', or 'change_pct'. Default 'market_cap'.

    Returns:
        Dict with: count, filters, results (sorted list of stocks with metrics).
    """
    return universe.screen(
        min_pe=min_pe,
        max_pe=max_pe,
        min_market_cap_egp_m=min_market_cap_egp_m,
        min_avg_volume=min_avg_volume,
        sector=sector,
        sort_by=sort_by,
    )


# ---------------------------------------------------------------------------
# INTELLIGENCE
# ---------------------------------------------------------------------------

@mcp.tool()
def get_disclosures(ticker: str | None = None, days: int = 7) -> dict[str, Any]:
    """Fetch official EGX disclosures from egx.com.eg.

    Disclosures are the canonical source for material events: dividends,
    capital actions, board changes, financial results, related-party
    transactions. Pulled directly from the EGX disclosures portal.

    Args:
        ticker: Optional EGX code to filter. If omitted, returns all
                market-wide disclosures.
        days: Lookback window in days. Default 7, max 90.

    Returns:
        Dict with: ticker, days, count, disclosures (list of {date, ticker,
        company, title_ar, title_en, url}).
    """
    return disclosures.fetch(ticker=ticker, days=min(days, 90))


@mcp.tool()
def get_news(ticker: str | None = None, lang: str = "en", limit: int = 10) -> dict[str, Any]:
    """Fetch recent news for an EGX security or the broader market.

    Combines headlines from Mubasher (Arabic) and Yahoo Finance (English).

    Args:
        ticker: Optional EGX code. If omitted, returns market-wide news.
        lang: 'ar' for Arabic, 'en' for English. Default 'en'.
        limit: Max articles. Default 10, max 30.

    Returns:
        Dict with: ticker, lang, count, articles (list of {date, source,
        title, url, summary}).
    """
    return news.fetch(ticker=ticker, lang=lang, limit=min(limit, 30))


# ---------------------------------------------------------------------------
# ANALYTICS
# ---------------------------------------------------------------------------

@mcp.tool()
def compute_technicals(ticker: str, period: str = "6mo") -> dict[str, Any]:
    """Compute standard technical indicators for an EGX security.

    Args:
        ticker: EGX code.
        period: History window. Default '6mo'.

    Returns:
        Dict with: ticker, as_of, price, indicators {
            rsi_14, macd, macd_signal, macd_histogram,
            sma_20, sma_50, sma_200,
            ema_20, ema_50,
            bb_upper, bb_middle, bb_lower,
            atr_14
        }, signals (human-readable interpretation).
    """
    return technicals.compute(ticker, period=period)


# ---------------------------------------------------------------------------
# PORTFOLIO
# ---------------------------------------------------------------------------

@mcp.tool()
def portfolio_summary(csv_path: str | None = None) -> dict[str, Any]:
    """Read a portfolio CSV and compute live P&L using current EGX prices.

    The CSV must have columns: ticker, shares, cost_basis (EGP per share).
    Optional columns: purchase_date, account, notes.

    If csv_path is omitted, reads from $EGX_PORTFOLIO_CSV (env var) or
    ~/egx_portfolio.csv as a fallback.

    Args:
        csv_path: Absolute path to CSV. Optional.

    Returns:
        Dict with: total_cost, total_value, total_pnl, total_pnl_pct,
        positions (list of {ticker, name, shares, cost_basis, current_price,
        market_value, unrealized_pnl, unrealized_pnl_pct, weight_pct}).
    """
    return portfolio.summary(csv_path=csv_path)


# ---------------------------------------------------------------------------
# DECISION LAYER — synthesize raw data into actionable verdicts
# ---------------------------------------------------------------------------

@mcp.tool()
def get_fundamentals(ticker: str) -> dict[str, Any]:
    """Return sanitized fundamentals: P/E, P/B, ROE, margin, leverage, dividend.

    Yahoo's `trailingPE` is unreliable for EGX names (CIRA notoriously
    returns 0.12). This tool corrects bogus P/E values by recomputing
    from price / trailing EPS, and flags whether a correction was made.

    Args:
        ticker: EGX code or nickname.

    Returns:
        Dict with: pe_ratio, pb_ratio, roe_pct, profit_margin_pct,
        debt_to_equity, dividend_yield_pct, market_cap, raw_pe_from_yahoo,
        pe_was_corrected.
    """
    return fundamentals.get_fundamentals(ticker)


@mcp.tool()
def get_macro_context() -> dict[str, Any]:
    """Return current macro snapshot relevant to EGX trading.

    EGP/USD spot, Brent crude, gold, CBE policy rates (deposit/lending),
    plus regime flags that classify the current macro setup (e.g. tight
    monetary regime → banks favored, real estate strained).

    Returns:
        Dict with: egp_usd, brent_usd, gold_usd, cbe_rates, regime_flags.
    """
    return macro.get_context()


@mcp.tool()
def score_stock(ticker: str) -> dict[str, Any]:
    """Compute the composite 0-100 score for a single stock.

    Four sub-scores weighted into a composite:
      - Valuation (30%): P/E and P/B vs. sector median, dividend yield
      - Quality   (25%): ROE, profit margin, debt/equity
      - Momentum  (25%): 6M return, RSI, MACD, MA50 vs MA200
      - Risk      (20%): annualized volatility, max drawdown

    A macro adjustment of ±5 points is applied based on sector fit with
    the current regime. Every sub-score returns its inputs and the points
    awarded so the verdict is fully auditable.

    Args:
        ticker: EGX code or nickname.

    Returns:
        Dict with: composite_score, subscores (with notes), sector_medians,
        macro_context, fundamentals_snapshot, weights.
    """
    return scoring.score_stock(ticker)


@mcp.tool()
def compare_peers(ticker: str, max_peers: int = 8) -> dict[str, Any]:
    """Rank a stock against its EGX sector peers on the same metrics.

    Returns target plus all sector peers with P/E, P/B, ROE, margin,
    dividend yield, and composite score side-by-side. Sorted by
    composite score so the best-of-sector floats to the top.

    Args:
        ticker: EGX code (must be in the curated universe).
        max_peers: Cap on results. Default 8.

    Returns:
        Dict with: target, sector, peer_count, target_rank_in_sector,
        target_relative_to_peers (best_in_sector / above_sector_average /
        in_line_with_sector / below_sector_average / worst_in_sector), peers.
    """
    return peers.compare(ticker, max_peers=max_peers)


@mcp.tool()
def position_size(
    ticker: str,
    portfolio_value_egp: float,
    risk_pct: float = 1.0,
    atr_multiple: float = 2.0,
    max_position_pct: float = 10.0,
) -> dict[str, Any]:
    """Compute share count, stop-loss, and target price for a planned trade.

    Uses ATR(14) for volatility-aware sizing. Default convention: risk 1%
    of portfolio with stop placed 2 ATRs below entry, target at 1:2 R/R,
    capped at 10% of portfolio per name.

    Args:
        ticker: EGX code.
        portfolio_value_egp: Total portfolio NAV in EGP.
        risk_pct: % of portfolio risked on this trade. Default 1.
        atr_multiple: Stop distance in ATRs. Default 2.
        max_position_pct: Cap on position weight. Default 10.

    Returns:
        Dict with: shares, position_cost_egp, position_weight_pct,
        stop_loss_price, stop_distance_egp, target_price, risk_egp,
        reward_to_risk, atr_14, method.
    """
    return sizing.position_size(
        ticker,
        portfolio_value_egp=portfolio_value_egp,
        risk_pct=risk_pct,
        atr_multiple=atr_multiple,
        max_position_pct=max_position_pct,
    )


@mcp.tool()
def get_catalyst_calendar(ticker: str, days_lookback: int = 14) -> dict[str, Any]:
    """Return forward catalysts: earnings, ex-dividend, disclosure clusters.

    Catalysts are the unscheduled price-movers that override technicals.
    Earnings within 7 days is a *blocking* signal — the decision layer
    will downgrade BUY → HOLD when this fires.

    Args:
        ticker: EGX code.
        days_lookback: Lookback window for disclosure clustering. Default 14.

    Returns:
        Dict with: next_earnings_date, days_to_earnings, ex_dividend_date,
        disclosure_count_window, recent_disclosures, catalyst_flags
        (with severity), blocking (bool).
    """
    return cal_mod.get_calendar(ticker, days_lookback=days_lookback)


@mcp.tool()
def decide(
    ticker: str,
    portfolio_value_egp: float | None = None,
    risk_pct: float = 1.0,
    round_trip_cost_pct: float = 1.0,
    min_net_edge_pct: float = 0.0,
) -> dict[str, Any]:
    """Synthesize all signals into a single actionable decision.

    Composes score + peer rank + catalyst calendar + macro + position
    sizing into one verdict, then applies a reliability gate:

        BUY        score ≥ 75, no blocking catalyst
        ACCUMULATE score 65-74
        HOLD       score 50-64 (or downgraded BUY: imminent earnings, or
                   cost-adjusted upside below threshold)
        REDUCE     score 35-49
        AVOID      score < 35
        ABSTAIN    data confidence too low to trust a valuation

    The gate (1) caps conviction to what the fundamentals support, (2) flags
    borderline scores and split sub-scores, (3) abstains when inputs can't be
    trusted, and (4) downgrades buy-side calls whose upside doesn't clear
    transaction costs. Check `actionable`, `data_confidence`,
    `expected_net_upside_pct`, and `abstain_reasons` in the output.

    Args:
        ticker: EGX code or nickname.
        portfolio_value_egp: Total portfolio NAV. Optional but enables sizing.
        risk_pct: Risk tolerance per trade. Default 1%.
        round_trip_cost_pct: Assumed round-trip cost (commission+fees+tax) for
            the net-edge calc. Default 1.0% — set to your broker's actual rate.
        min_net_edge_pct: Min net upside after cost to keep a buy-side verdict.

    Returns:
        Dict with: verdict, conviction, actionable, abstain_reasons,
        data_confidence, expected_net_upside_pct, composite_score,
        fair_value_estimate, suggested_levels, key_drivers, key_risks,
        blocking_catalysts, subscores, peer_relative, macro_bias,
        days_to_earnings, data_quality_notes, disclaimer.
    """
    return decision.decide(
        ticker,
        portfolio_value_egp=portfolio_value_egp,
        risk_pct=risk_pct,
        round_trip_cost_pct=round_trip_cost_pct,
        min_net_edge_pct=min_net_edge_pct,
    )


# ---------------------------------------------------------------------------
# SHORT-TERM SIMULATION — bootstrap MC + technical edge overlay
# ---------------------------------------------------------------------------

@mcp.tool()
def simulate_short_term(
    ticker: str,
    horizon_days: int = 5,
    n_paths: int = 2000,
    lookback_days: int = 60,
) -> dict[str, Any]:
    """Probabilistic 1–10 day forecast for one stock (bootstrap MC).

    Resamples the trailing daily-return distribution with replacement
    `n_paths` times, applies a technical edge overlay (RSI / trend /
    MACD / Bollinger that shifts daily drift in ±0.6%), and returns the
    full empirical distribution of terminal prices.

    Args:
        ticker: EGX code or nickname.
        horizon_days: Forecast window. Default 5.
        n_paths: Number of MC paths. Default 2000.
        lookback_days: Days of history feeding the bootstrap. Default 60.

    Returns:
        Dict with: current_price, expected_return_pct, p10/p50/p90 terminal
        prices, prob_up_2pct, prob_up_5pct, prob_down_2pct, prob_down_5pct,
        edge_drift_pct_per_day, edge_drivers, imminent_move_score, method.
    """
    return simulation.simulate_one(
        ticker,
        horizon_days=horizon_days,
        n_paths=n_paths,
        lookback_days=lookback_days,
    )


@mcp.tool()
def scan_short_term_winners(
    horizon_days: int = 5,
    n_paths: int = 2000,
    lookback_days: int = 60,
    min_prob_up_2pct: float = 0.5,
    min_expected_return_pct: float = 0.0,
    full_market: bool = True,
) -> dict[str, Any]:
    """Run the short-term simulator across the EGX universe and rank.

    For each name in the universe, runs `simulate_short_term`, filters
    by minimum upside probability and expected return, and ranks by
    `imminent_move_score = 100 × E[r] × P(up>2%) / σ(terminal_r)`.

    Args:
        horizon_days: Forecast window. Default 5.
        n_paths: Paths per ticker. Default 2000.
        lookback_days: History window. Default 60.
        min_prob_up_2pct: Filter — minimum P(>2% gain). Default 0.5.
        min_expected_return_pct: Filter — minimum expected return. Default 0.
        full_market: If True (default), scans the validated extended
            EGX universe (~70 names). If False, only the 29 curated names.

    Returns:
        Dict with: top_5 spotlight, full ranked list, method, skipped tickers.
    """
    return simulation.scan_universe(
        horizon_days=horizon_days,
        n_paths=n_paths,
        lookback_days=lookback_days,
        min_prob_up_2pct=min_prob_up_2pct,
        min_expected_return_pct=min_expected_return_pct,
        full_market=full_market,
    )


# ---------------------------------------------------------------------------
# PORTFOLIO MANAGER LAYER — risk, optimization, factor exposure, backtest
# ---------------------------------------------------------------------------

@mcp.tool()
def get_egp_risk_free_rate() -> dict[str, Any]:
    """Return the EGP T-bill rate used as the excess-return baseline.

    Source priority: env override → CBE T-bill auction scrape → CBE
    corridor proxy → hardcoded fallback. Used everywhere the model
    computes excess returns or Sharpe-style scores.
    """
    return risk_free.get_rate()


@mcp.tool()
def detect_market_regime() -> dict[str, Any]:
    """Classify the current EGX regime (BULL / BEAR / HIGH_VOL / SIDEWAYS).

    Returns the regime label, the metrics that drove the classification,
    and the per-regime weight overrides the scoring engine applies. The
    same composite score will weight momentum more in BULL regimes and
    quality/risk more in HIGH_VOL regimes.
    """
    return regime.classify()


@mcp.tool()
def check_liquidity(
    ticker: str,
    intended_shares: int,
    max_participation_pct: float = 15.0,
) -> dict[str, Any]:
    """Verify a planned order against ADV. Returns slippage estimate.

    Args:
        ticker: EGX code.
        intended_shares: Shares planned.
        max_participation_pct: Cap on % of ADV. Default 15.

    Returns:
        Dict with: feasible, participation_pct_of_adv, max_safe_shares,
        estimated_slippage_bps, estimated_slippage_egp, recommendation.
    """
    return liquidity.check_capacity(
        ticker,
        intended_shares=intended_shares,
        max_participation_pct=max_participation_pct,
    )


@mcp.tool()
def portfolio_factor_exposure(
    tickers: list[str],
    weights: list[float] | None = None,
    lookback_days: int = 90,
) -> dict[str, Any]:
    """Decompose a portfolio's return drivers across macro factors.

    Regresses each ticker on EGX 30, USDEGP, Brent, Gold, EM equities,
    then aggregates with portfolio weights. Tells you whether you're
    *unintentionally* long oil, short EGP, or just market beta.

    Args:
        tickers: List of EGX codes.
        weights: Optional list of weights (defaults to equal-weight).
        lookback_days: Regression window. Default 90.
    """
    return factors.portfolio_factor_exposure(
        tickers, weights=weights, lookback_days=lookback_days,
    )


@mcp.tool()
def portfolio_risk(
    tickers: list[str],
    weights: list[float] | None = None,
    lookback_days: int = 252,
    confidence: float = 0.95,
    horizon_days: int = 1,
    nav_egp: float | None = None,
) -> dict[str, Any]:
    """Compute portfolio-level VaR, CVaR, drawdown, Sharpe.

    Historical-simulation VaR (no parametric assumptions). Includes
    a circuit-breaker check that fires when rolling 20-day drawdown
    breaches –8%. Sharpe is excess over the live EGP T-bill rate.

    Args:
        tickers: List of EGX codes.
        weights: Portfolio weights (defaults to equal-weight).
        lookback_days: History window for the loss distribution. Default 252.
        confidence: VaR confidence level. Default 0.95.
        horizon_days: VaR horizon (sqrt-time scaled). Default 1.
        nav_egp: Optional NAV — if provided, returns VaR/CVaR in EGP.
    """
    return risk_mod.portfolio_risk(
        tickers,
        weights=weights,
        lookback_days=lookback_days,
        confidence=confidence,
        horizon_days=horizon_days,
        nav_egp=nav_egp,
    )


@mcp.tool()
def optimize_portfolio(
    tickers: list[str],
    method: str = "min_variance",
    expected_returns_pct: dict[str, float] | None = None,
    target_vol_pct: float | None = None,
    max_weight: float | None = 0.20,
    nav_egp: float | None = None,
    lookback_days: int = 252,
) -> dict[str, Any]:
    """Compute optimal portfolio weights.

    Methods:
        equal_weight   sanity baseline
        min_variance   minimize portfolio vol (closed-form)
        risk_parity    equalize risk contributions (iterative)
        tangency       maximum Sharpe — needs expected_returns_pct

    Args:
        tickers: Candidate basket.
        method: Allocation method.
        expected_returns_pct: Required for tangency. Annual %, e.g. {"COMI": 18.5}.
        target_vol_pct: Optional cap on annualized vol. Excess scaled to cash.
        max_weight: Per-name cap. Default 0.20 (20%).
        nav_egp: Optional NAV — returns EGP allocations.
        lookback_days: History for covariance. Default 252.
    """
    return optimizer.optimize(
        tickers,
        method=method,
        expected_returns_pct=expected_returns_pct,
        target_vol_pct=target_vol_pct,
        max_weight=max_weight,
        nav_egp=nav_egp,
        lookback_days=lookback_days,
    )


@mcp.tool()
def backtest_strategy(
    start: str = "2023-01-01",
    end: str | None = None,
    top_n: int = 5,
    rebalance_days: int = 21,
    universe: str = "extended",
    min_roe_pct: float | None = 10.0,
) -> dict[str, Any]:
    """Walk-forward backtest of the V8b production strategy.

    V8b = price-momentum score (V3) + ROE quality pre-filter. The filter
    drops any name with ROE below `min_roe_pct` from the candidate pool
    BEFORE ranking. Set min_roe_pct=None to revert to V3 (no filter).

    Each rebalance: score every name using only data available at that
    date, take top_n equal-weight, hold for one period. Aggregates into
    Sharpe, max DD, hit rate, and IR vs EGX 30.

    Args:
        start: ISO start date.
        end: ISO end date. Defaults to today.
        top_n: Names held per period. Default 5.
        rebalance_days: Period length in trading days. Default 21 (~monthly).
        universe: 'extended' (~70 names) or 'curated' (29).
        min_roe_pct: Quality filter threshold. Default 10.0. Set None to disable.
    """
    return backtest.backtest(
        start=start, end=end, top_n=top_n,
        rebalance_days=rebalance_days, universe=universe,
        min_roe_pct=min_roe_pct,
    )


# ---------------------------------------------------------------------------
# WEEKLY TRADING MODEL (W1) — short-horizon (5-day) ranker
# ---------------------------------------------------------------------------

@mcp.tool()
def weekly_top_picks(asof: str | None = None, top_n: int = 5) -> dict[str, Any]:
    """Today's top weekly trading picks (5-day horizon).

    The W1 model was tuned on 2024-2026 data and walk-forward validated.
    Across all tested windows it beat the broad EGX market by 44-85pp
    annualized with Sharpe 2-4× higher than passive.

    Validation summary:
      Full 28mo:    +107.8% CAGR, Sharpe 1.64  (vs market +40.8%, 0.65)
      Holdout OOS:  +131.0% CAGR, Sharpe 2.78  (vs market +50.1%, 1.32)

    Score formula (winning config from random search):
      1.5×mom_5d + 0.5×(-mom_1d) + 3×breakout_signal + 3×above_MA20
      − 0.3×stretched_5d + 5×dip_in_uptrend
      filtered by ROE≥10% and volume≥50% of ADV.

    Args:
        asof: ISO date. Defaults to today.
        top_n: Names to return. Default 5.

    Returns:
        Dict with: top_picks (with score, drivers, momentum decomposition),
        runners_up, n_eligible, validation_summary.
    """
    return weekly.rank_universe(asof=asof, top_n=top_n)


# ---------------------------------------------------------------------------
# AGENTIC LAYER — analyst aliases, debate, risk gate, reflection, trader plan
# ---------------------------------------------------------------------------
# Inspired by TradingAgents' multi-tier architecture (analyst → researcher
# debate → risk manager → portfolio manager) and the llm-council
# three-stage pattern, but executed deterministically inside the MCP so
# the host LLM (Claude) gets a structured dossier instead of a single
# verdict to either trust or override blindly.

@mcp.tool()
def fundamentals_analyst(ticker: str) -> dict[str, Any]:
    """Analyst-team alias for `get_fundamentals`. Sanitized P/E, P/B, ROE,
    margin, leverage, dividend yield with audit trail.
    """
    return fundamentals.get_fundamentals(ticker)


@mcp.tool()
def technical_analyst(ticker: str, period: str = "6mo") -> dict[str, Any]:
    """Analyst-team alias for `compute_technicals`. RSI, MACD, MAs,
    Bollinger, ATR + signal interpretation.
    """
    return technicals.compute(ticker, period=period)


@mcp.tool()
def news_analyst(ticker: str | None = None, lang: str = "en", limit: int = 10) -> dict[str, Any]:
    """Analyst-team alias for `get_news`. Mubasher (AR) + Yahoo (EN)
    headlines, with disclosure context.
    """
    return news.fetch(ticker=ticker, lang=lang, limit=min(limit, 30))


@mcp.tool()
def sentiment_analyst(
    ticker: str | None = None,
    lang: str = "both",
    limit: int = 15,
    backend: str | None = None,
) -> dict[str, Any]:
    """Score recent EGX headlines as bullish / bearish.

    Default backend is a finance-tuned AR+EN lexicon (no model, no API key).
    Set backend='transformer' (or 'auto') to use FinBERT (EN) + CAMeLBERT-DA
    (AR) when the optional deps are installed (`pip install egx-mcp[sentiment]`);
    it falls back to the lexicon per-language if a model can't load.

    Args:
        ticker: EGX code or nickname. Omit for market-wide.
        lang: 'en', 'ar', or 'both'. Default 'both'.
        limit: Max headlines per language. Default 15.
        backend: 'lexicon' (default), 'transformer', or 'auto'. Omit to use
            the EGX_SENTIMENT_BACKEND env var.

    Returns:
        Dict with: aggregate_score (-1..+1), label, coverage_pct,
        bull_signals, bear_signals, per-headline scores, backend (effective).
    """
    return sentiment.analyze_sentiment(ticker, lang=lang, limit=limit, backend=backend)


@mcp.tool()
def tag_disclosure_events(ticker: str | None = None, days: int = 14) -> dict[str, Any]:
    """Tag recent EGX disclosures into price-moving event types (dividend,
    capital increase, M&A, profit warning, earnings, board change, trading
    suspension) using a multilingual zero-shot classifier — AR + EN.

    Falls back to a keyword classifier if the model isn't installed
    (`pip install egx-mcp[sentiment]`). Returns each disclosure tagged with
    event + confidence + a `material` flag, plus a `material_events`
    shortlist the decision layer can gate a BUY on.

    Args:
        ticker: EGX code or nickname. Omit for market-wide disclosures.
        days: Lookback window. Default 14.
    """
    return events_mod.tag_disclosures(ticker, days=days)


@mcp.tool()
def ir_archive_status(ticker: str | None = None) -> dict[str, Any]:
    """Report the investor-relations document archive for one company or all.

    Returns the discovered IR url, document count, and last-fetch time per
    company. Read-only — the fetch/discover/extract/promote pipeline runs as
    `python -m scripts.ir_pipeline` (network + human review gates), not as a
    tool, so nothing is downloaded or promoted into fundamentals unattended.

    Args:
        ticker: EGX code/nickname for one company, or omit for the whole archive.
    """
    return ir_fetch_mod.status(ticker)


@mcp.tool()
def get_ir_context(ticker: str) -> dict[str, Any]:
    """Surface a company's archived investor-relations material as decision
    context — document list + provisional figures, each with its source page
    and the exact text snippet it came from.

    Read this when forming a verdict on a name to factor in what the company
    itself published (financial statements, earnings releases, presentations).
    These figures are UNVERIFIED decision context, not scored inputs — cite
    and sanity-check them; they only enter the quantitative score via the
    explicit promote gate. Empty until you run the IR pipeline
    (`python -m scripts.ir_pipeline fetch/extract`).

    Args:
        ticker: EGX code or nickname.
    """
    return ir_extract_mod.company_context(ticker)


@mcp.tool()
def forecast_price(ticker: str, horizon_days: int = 21) -> dict[str, Any]:
    """Probabilistic forward-return forecast for an EGX name via a zero-shot
    time-series foundation model (Chronos).

    Returns expected_return_pct and uncertainty_pct (q90-q10 spread, a
    model-implied volatility). Falls back to a naive drift+vol estimate if
    the model isn't installed (`pip install egx-mcp[forecast]`).

    One input among many — EGX is thin and FX-driven, so backtest before
    trusting it. A wide uncertainty band argues for sizing down.

    Args:
        ticker: EGX code or nickname.
        horizon_days: Forecast horizon in trading days. Default 21 (~1 month).
    """
    return forecast_mod.forecast_return(ticker, horizon_days=horizon_days)


@mcp.tool()
def macro_analyst() -> dict[str, Any]:
    """Analyst-team alias for `get_macro_context`. EGP/USD, Brent, gold,
    CBE rates + per-sector regime bias.
    """
    return macro.get_context()


@mcp.tool()
def debate_ticker(ticker: str, include_sentiment: bool = True) -> dict[str, Any]:
    """Run the bull / bear / chairman three-stage debate on one EGX name.

    A bull researcher and bear researcher each pull weighted theses from
    scoring, peers, calendar, and sentiment. The chairman weighs net edge
    and produces a synthesis verdict + conviction.

    The chairman is rule-based, not an LLM — but the structured dossier
    is exactly what the host LLM needs to override or extend.

    Args:
        ticker: EGX code or nickname.
        include_sentiment: Pull headline sentiment too. Default True.

    Returns:
        Dict with: composite_score, bull_case, bear_case, chairman
        (verdict, conviction, edge, rationale, deciding_factors),
        sentiment_summary.
    """
    return debate_mod.debate(ticker, include_sentiment=include_sentiment)


@mcp.tool()
def risk_gate_review(
    ticker: str,
    proposed_verdict: str,
    suggested_levels: dict[str, Any] | None = None,
    portfolio_csv: str | None = None,
    constraints: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Portfolio-aware veto check on a proposed verdict.

    Applies single-name cap, sector cap, position-count cap, drawdown
    circuit breaker, and post-add portfolio beta. Either passes the
    verdict through or downgrades it (BUY → ACCUMULATE → HOLD) when a
    constraint would breach.

    Args:
        ticker: EGX code or nickname.
        proposed_verdict: BUY / ACCUMULATE / HOLD / REDUCE / AVOID / WAIT.
        suggested_levels: Optional sizing block from decide() — used to
            compute the candidate's NAV weight. If None, assumes 5%.
        portfolio_csv: Path to portfolio CSV. Defaults to env / home.
        constraints: Override defaults — keys: max_single_name_pct,
            max_sector_pct, max_position_count, min_cash_pct,
            drawdown_circuit_breaker, max_portfolio_beta.

    Returns:
        Dict with: original_verdict, final_verdict, downgrades (reasons),
        breaches (raw checks), portfolio_snapshot.
    """
    return risk_gate_mod.risk_gate(
        ticker,
        proposed_verdict=proposed_verdict,
        suggested_levels=suggested_levels,
        portfolio_csv=portfolio_csv,
        constraints=constraints,
    )


@mcp.tool()
def trader_plan(
    ticker: str,
    portfolio_value_egp: float,
    risk_pct: float = 1.0,
    atr_multiple: float = 2.0,
    max_position_pct: float = 10.0,
    tranches: int = 3,
) -> dict[str, Any]:
    """Tranched entry / scale-out plan around the sized position.

    For thin EGX names where a single fill pays slippage. Returns:
      - Entry tranche 1: starter at market
      - Entry tranche 2: pullback to MA20 or entry−1 ATR
      - Entry tranche 3: breakout +0.5 ATR with RSI confirmation
      - Exit 1: +1 ATR — book 1/3, stop to breakeven
      - Exit 2: +2 ATR — book 1/3
      - Exit 3: chandelier trail (running high − 3 ATR) for the rest

    Args:
        ticker: EGX code or nickname.
        portfolio_value_egp: NAV.
        risk_pct: Per-trade risk. Default 1%.
        atr_multiple: Stop in ATRs. Default 2.
        max_position_pct: Cap. Default 10.
        tranches: 1, 2, or 3 entry slices. Default 3.

    Returns:
        Dict with: entry_plan (per tranche), exit_plan (per level),
        stop_loss_price, total_shares, base_sizing.
    """
    return sizing.trader_plan(
        ticker,
        portfolio_value_egp=portfolio_value_egp,
        risk_pct=risk_pct,
        atr_multiple=atr_multiple,
        max_position_pct=max_position_pct,
        tranches=tranches,
    )


@mcp.tool()
def log_decision(
    ticker: str,
    verdict: str,
    composite_score: float | None = None,
    fair_value: float | None = None,
    target_price: float | None = None,
    stop_loss: float | None = None,
    entry_price: float | None = None,
    conviction: str | None = None,
    blocking_catalysts: list[str] | None = None,
    tags: list[str] | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Append a decision record to the persistent decision log (JSONL).

    Used to build the memory the reflection tool replays. Idempotent and
    append-only. Default path: <repo>/logs/decisions.jsonl, overridable
    via $EGX_DECISION_LOG.

    Tags help reflection cluster failures — useful tags include the
    sector, regime label, "earnings_within_7d", "macro_headwind",
    "high_conviction".
    """
    return reflection.log_decision(
        ticker=ticker,
        verdict=verdict,
        composite_score=composite_score,
        fair_value=fair_value,
        target_price=target_price,
        stop_loss=stop_loss,
        entry_price=entry_price,
        conviction=conviction,
        blocking_catalysts=blocking_catalysts,
        tags=tags,
        note=note,
    )


@mcp.tool()
def backtest_agentic(
    start: str = "2023-01-01",
    end: str | None = None,
    rebalance_days: int = 21,
    universe: str = "extended",
    buy_only_top_n: int | None = 5,
    include_accumulate: bool = True,
) -> dict[str, Any]:
    """Walk-forward backtest of the chairman's verdict bands.

    Maps the V3 price-only score (point-in-time clean) into chairman
    bands each period, then:
      - grades forward returns by band (BUY / ACCUMULATE / HOLD /
        REDUCE / AVOID) — tests whether the mapping has edge
      - simulates a BUY-side equal-weight portfolio vs EGX 30

    Caveat: only the price-momentum half of the live chairman is replayed
    (fundamentals / sentiment / calendar are not point-in-time
    reconstructable from current data sources). Use alongside
    `backtest_strategy` for the full V8b production picture.

    Args:
        start, end: ISO window.
        rebalance_days: ~21 = monthly. Default 21.
        universe: 'extended' (~70) or 'curated' (29).
        buy_only_top_n: Cap on BUY+ACCUMULATE portfolio per period.
            Default 5. None = uncapped.
        include_accumulate: Include ACCUMULATE band in the BUY portfolio.
            Default True.

    Returns:
        Dict with: by_verdict (n, hit_rate, mean_return per band),
        monotonicity_check (BUY > ACCUM > HOLD > REDUCE > AVOID?),
        buy_portfolio summary, benchmark_egx30, period samples.
    """
    return agentic_backtest.backtest_agentic(
        start=start,
        end=end,
        rebalance_days=rebalance_days,
        universe=universe,
        buy_only_top_n=buy_only_top_n,
        include_accumulate=include_accumulate,
    )


@mcp.tool()
def reflect_on_decisions(
    window_days: int = 90,
    hold_days: int = 21,
) -> dict[str, Any]:
    """Replay the decision log against realized returns and grade hits.

    For each logged decision in the window, fetches close-to-close
    return over `hold_days`, then grades:
        BUY/ACCUMULATE  hit if return > 0
        REDUCE/AVOID    hit if return < 0
        HOLD/WAIT       hit if |return| < 5%

    Returns aggregate hit rate, mean return, breakdown by verdict and
    tag, recent worst misses, and surfaced lessons (e.g. "BUYs with
    blocking_catalyst tag hit 25% over 8 samples — tighten the filter").

    Args:
        window_days: How far back to grade. Default 90.
        hold_days: Holding period applied to each call. Default 21.
    """
    return reflection.reflect(window_days=window_days, hold_days=hold_days)


# ---------------------------------------------------------------------------
# COMPANY BRIEF
# ---------------------------------------------------------------------------

@mcp.tool()
def company_brief_full(
    ticker: str,
    portfolio_value_egp: float | None = None,
) -> dict[str, Any]:
    """Comprehensive intelligence brief on a single EGX name.

    Aggregates every data source the MCP can reach into one structured
    snapshot — quote, fundamentals (audited from Mubasher), technicals,
    monthly V8b decision, weekly W1 rank, peer position, factor exposures,
    macro fit, catalysts, recent news, 5-day Monte Carlo forecast.

    Designed for the 30-second pre-trade check: read the summary_for_pm
    block first, drill into specific sections only as needed.

    Args:
        ticker: EGX code or nickname.
        portfolio_value_egp: Optional NAV — enables sized-position output
            in the monthly_decision_v8b block.

    Returns:
        Structured dict with: snapshot, fundamentals, technicals,
        weekly_model_w1, monthly_decision_v8b, short_term_simulation_5d,
        peer_context, factor_exposures_90d, macro_fit, catalysts,
        recent_news, summary_for_pm.
    """
    return company_brief.brief(ticker, portfolio_value_egp=portfolio_value_egp)


# ---------------------------------------------------------------------------
# BEHAVIOR DECOMPOSITION — what drives each stock?
# ---------------------------------------------------------------------------

@mcp.tool()
def stock_behavior_profile(ticker: str, lookback_days: int = 120) -> dict[str, Any]:
    """Explain what *drives* a single EGX stock's price behavior.

    Combines four lenses into one profile:
      - Macro factor betas: sensitivity to EGX30 (market), EGP/USD, Brent
        oil, gold, and EM equities.
      - Systematic vs idiosyncratic share: R² of the factor model. High R²
        means the name is macro-driven; low R² means it moves on its own
        (news / disclosures / flows).
      - Risk character: annualized volatility, max drawdown, trailing return.
      - Fundamental anchor: P/E, P/B, ROE, leverage, dividend yield.

    Args:
        ticker: EGX code or nickname.
        lookback_days: Factor-regression and risk window. Default 120.

    Returns:
        Dict with: ticker, name, sector, drivers (market_beta, factor_betas,
        dominant_macro_factor, systematic_r2, idiosyncratic_pct, alpha),
        risk, fundamentals, interpretation (plain-English notes).
    """
    return behavior.stock_behavior(ticker, lookback_days=lookback_days)


@mcp.tool()
def scan_universe_behavior(
    universe: str = "extended",
    lookback_days: int = 120,
    sector: str | None = None,
) -> dict[str, Any]:
    """Profile every reachable EGX name and roll the drivers up by sector.

    Runs `stock_behavior_profile` across the universe, then aggregates each
    sector's average market beta, dominant macro factor, average volatility,
    and average idiosyncratic share — so you can see which forces move each
    corner of the market.

    Coverage: EGX has ~240 listings but Yahoo Finance only carries reliable
    daily history for the validated extended set (~68 names). Names without
    usable history are reported under `skipped`, not silently dropped.

    Args:
        universe: 'extended' (~68 validated) or 'curated' (~30 named).
        lookback_days: Regression/risk window. Default 120.
        sector: Optional sector filter (case-insensitive substring), e.g.
            'Banks', 'Real Estate', 'Chemicals'.

    Returns:
        Dict with: as_of, universe, n_scanned, n_with_data, n_skipped,
        by_sector (per-sector roll-up), stocks (per-name profiles),
        skipped, coverage_note.
    """
    return behavior.scan_universe_behavior(
        universe=universe,
        lookback_days=lookback_days,
        sector=sector,
    )


@mcp.tool()
def refresh_price_cache(
    universe: str = "extended",
    lookback_days: int = 400,
) -> dict[str, Any]:
    """Rebuild the offline price + driver cache from investing.com.

    Pulls daily OHLCV for the whole universe plus the five macro factors
    (EGX30, USD/EGP, Brent, gold, EM equities), computes each name's driver
    profile (factor betas, R², volatility, drawdown, momentum), and writes
    it all to disk. After this runs, the behavior/quote tools read prices
    and drivers offline — they keep working even when the live feed is down.

    This is a heavy, network-bound call (one request per ticker, throttled).
    Run it once to populate the cache, then re-run periodically to refresh.

    Args:
        universe: 'extended' (full validated set) or 'curated' (~30 named).
        lookback_days: History window to pull and regress over. Default 400.

    Returns:
        Dict with: status, cache_path, refreshed_at, date_range, n_tickers,
        n_failed, failed.
    """
    return price_cache.refresh(universe=universe, lookback_days=lookback_days)


@mcp.tool()
def price_cache_status() -> dict[str, Any]:
    """Report whether the offline price cache is populated and how fresh it is.

    Returns availability, last refresh timestamp, covered date range, ticker
    count, and the universe it was built from.
    """
    return price_cache.meta()


@mcp.tool()
def cached_quote(ticker: str) -> dict[str, Any]:
    """Latest cached daily bar for an EGX ticker (offline, from price cache).

    Returns price, previous_close, change_pct, date, volume. If the cache is
    empty or the ticker isn't covered, returns an error field — run
    `refresh_price_cache` first.
    """
    return price_cache.get_quote(ticker)


@mcp.tool()
def render_company_briefing(ticker: str, out_dir: str | None = None) -> dict[str, Any]:
    """Render a self-contained HTML briefing page for one EGX name.

    Produces a single .html file (open in any browser, no network needed):
    a candlestick + volume chart drawn from the cached price history, plus
    panels for the driver decomposition (factor betas, R², volatility,
    drawdown, momentum, interpretation), fundamentals, and the monthly
    verdict. The TradingView Lightweight Charts library is inlined, and the
    chart/drivers come from the offline price cache — so the page renders
    even when the live feed is down. Run `refresh_price_cache` first if the
    ticker isn't cached yet.

    Args:
        ticker: EGX ticker (e.g. 'TMGH', 'COMI').
        out_dir: Optional output directory. Defaults to egx_mcp/data/briefings.

    Returns:
        Dict with: status, ticker, name, path (the .html file), bars,
        date_range, has_verdict. On a cache miss, returns an error field.
    """
    return briefing_page.render(ticker, out_dir=out_dir)


@mcp.tool()
def project_impact(ticker: str, limit: int = 25) -> dict[str, Any]:
    """Estimate how future-project news may move a stock, via event study.

    Two halves glued together:
      1. An offline event study on the cached price history. Using the name's
         factor model (alpha + betas on EGX30/EGP/Brent/gold/EM), the macro
         component is stripped from every daily return to leave the *abnormal
         return* — the stock-specific move. Its distribution calibrates the
         "typical catalyst move" (90th-pct abnormal return) with a low/high
         band, plus the biggest historical event days.
      2. A live catalyst scan of recent headlines, filtered to project /
         contract / land / capital items and scored for tone. Each material
         item's expected impact is direction (tone) × magnitude (the typical
         catalyst move). Degrades gracefully to profile-only when offline.

    Output is a statistical reaction estimate, not a forecast. Run
    `refresh_price_cache` first if the ticker isn't cached.

    Args:
        ticker: EGX ticker (e.g. 'TMGH', 'COMI').
        limit: Max headlines per language to scan. Default 25.

    Returns:
        Dict with: reaction_profile (event-study stats), catalyst_scan
        (scored project headlines + net expected move), interpretation,
        method, disclaimer. On a cache miss, returns an error field.
    """
    return project_impact_mod.project_impact(ticker, limit=limit)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Starting EGX MCP server (stdio transport)")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
