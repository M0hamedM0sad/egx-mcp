"""Run baseline + agentic backtests and dump results to JSON.

Used for the model-vs-actual comparison report. The two functions are
independent so we run them sequentially for cleaner logs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Quiet the noisy adapters; we just want the final results
logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

from egx_mcp.data import backtest as bt_mod
from egx_mcp.data import agentic_backtest as ab_mod

OUT = Path(__file__).resolve().parent.parent / "logs"
OUT.mkdir(parents=True, exist_ok=True)


def run(name: str, fn, **kw):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    print(f"args: {kw}", flush=True)
    try:
        result = fn(**kw)
    except Exception as e:
        result = {"error": str(e)}
    dt = time.time() - t0
    out_path = OUT / f"backtest_{name}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  done in {dt:.1f}s -> {out_path}", flush=True)
    return result


if __name__ == "__main__":
    # 1. V8b baseline (production strategy already shipped)
    base = run(
        "v8b_baseline",
        bt_mod.backtest,
        start="2024-01-01",
        end=None,
        top_n=5,
        rebalance_days=21,
        universe="curated",
        min_roe_pct=10.0,
    )

    # 2. Agentic verdict-band backtest (new module)
    agentic = run(
        "agentic_curated",
        ab_mod.backtest_agentic,
        start="2024-01-01",
        end=None,
        rebalance_days=21,
        universe="curated",
        buy_only_top_n=5,
        include_accumulate=True,
    )

    # 3. Quick highlights to stdout
    print("\n=== HIGHLIGHTS ===", flush=True)
    print(f"V8b baseline           total: {base.get('total_return_pct')}%  "
          f"CAGR: {base.get('annualized_return_pct')}%  "
          f"Sharpe: {base.get('annualized_sharpe')}  "
          f"DD: {base.get('max_drawdown_pct')}%  "
          f"vs EGX30 alpha: {base.get('alpha_vs_benchmark_pct')}%", flush=True)

    bp = (agentic.get("buy_portfolio") or {})
    bn = (agentic.get("benchmark_egx30") or {})
    print(f"Agentic BUY portfolio  total: {bp.get('total_return_pct')}%  "
          f"CAGR: {bp.get('annualized_return_pct')}%  "
          f"Sharpe: {bp.get('annualized_sharpe')}  "
          f"DD: {bp.get('max_drawdown_pct')}%  "
          f"vs EGX30 alpha: {bn.get('alpha_pct')}%", flush=True)

    mc = agentic.get("monotonicity_check", {})
    print(f"\nMonotonicity: {mc.get('monotone_decreasing')}  "
          f"BUY-AVOID spread: {mc.get('buy_minus_avoid_pct')}%", flush=True)
    if mc.get("ordered_means"):
        for v, mu in mc["ordered_means"]:
            n = (agentic.get("by_verdict") or {}).get(v, {}).get("n", 0)
            hr = (agentic.get("by_verdict") or {}).get(v, {}).get("hit_rate_pct", 0)
            print(f"  {v:<12s} mean: {mu:+.3f}%  hit_rate: {hr}%  n={n}", flush=True)
