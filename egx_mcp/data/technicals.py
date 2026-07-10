"""Technical indicators — pure pandas, no TA-Lib dependency.

Indicators implemented:
  - RSI (14)
  - MACD (12, 26, 9)
  - Simple MAs (20, 50, 200)
  - Exponential MAs (20, 50)
  - Bollinger Bands (20, 2)
  - ATR (14)
"""
from __future__ import annotations

import logging
from typing import Any

from .universe import resolve_ticker
from . import investing

log = logging.getLogger("egx-mcp.technicals")

_PERIOD_DAYS = {"1mo": 30, "3mo": 95, "6mo": 190, "1y": 370, "2y": 740}


def compute(user_ticker: str, period: str = "6mo") -> dict[str, Any]:
    canonical, yahoo, name = resolve_ticker(user_ticker)
    # Glitch-guarded source (investing.com primary, zero-volume bars dropped).
    # Always pull >=260 trading days so SMA-200 is available regardless of period.
    lookback = max(_PERIOD_DAYS.get(period, 260), 260)
    df = investing.daily_history(canonical, lookback_days=lookback)

    if df.empty or len(df) < 30:
        return {
            "ticker": canonical,
            "yahoo_symbol": yahoo,
            "error": f"Insufficient data for technicals (got {len(df)} bars).",
        }

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Moving averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean() if len(close) >= 200 else None
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = tr1.combine(tr2, max).combine(tr3, max)
    atr = tr.rolling(14).mean()

    last = -1
    price = float(close.iloc[last])

    def _f(series, idx=last) -> float | None:
        try:
            v = float(series.iloc[idx])
            return round(v, 4) if v == v else None  # filter NaN
        except (IndexError, TypeError, AttributeError):
            return None

    indicators = {
        "rsi_14": _f(rsi),
        "macd": _f(macd_line),
        "macd_signal": _f(macd_signal),
        "macd_histogram": _f(macd_hist),
        "sma_20": _f(sma_20),
        "sma_50": _f(sma_50),
        "sma_200": _f(sma_200) if sma_200 is not None else None,
        "ema_20": _f(ema_20),
        "ema_50": _f(ema_50),
        "bb_upper": _f(bb_upper),
        "bb_middle": _f(bb_mid),
        "bb_lower": _f(bb_lower),
        "atr_14": _f(atr),
    }

    # Generate plain-language signals
    signals = []
    rsi_val = indicators["rsi_14"]
    if rsi_val is not None:
        if rsi_val > 70:
            signals.append(f"RSI {rsi_val:.1f}: overbought")
        elif rsi_val < 30:
            signals.append(f"RSI {rsi_val:.1f}: oversold")
        else:
            signals.append(f"RSI {rsi_val:.1f}: neutral")

    macd_v = indicators["macd"]
    macd_s = indicators["macd_signal"]
    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s:
            signals.append("MACD above signal: bullish momentum")
        else:
            signals.append("MACD below signal: bearish momentum")

    sma50 = indicators["sma_50"]
    sma200 = indicators["sma_200"]
    if sma50 and sma200:
        if sma50 > sma200:
            signals.append("Golden cross territory (SMA50 > SMA200)")
        else:
            signals.append("Death cross territory (SMA50 < SMA200)")

    if indicators["bb_upper"] and price > indicators["bb_upper"]:
        signals.append("Price above upper Bollinger Band — extended")
    elif indicators["bb_lower"] and price < indicators["bb_lower"]:
        signals.append("Price below lower Bollinger Band — extended")

    return {
        "ticker": canonical,
        "yahoo_symbol": yahoo,
        "name": name,
        "as_of": df.index[-1].strftime("%Y-%m-%d"),
        "price": round(price, 4),
        "indicators": indicators,
        "signals": signals,
    }
