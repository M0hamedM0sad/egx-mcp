"""MCP transport reliability validation — does every tool honour its contract?

The statistical harness (tests/validate_edge.py) asks "is the model's edge
real?". This asks the prerequisite question: "does the MCP server itself behave
reliably?" — because a verdict you can't trust the transport to deliver is
worthless regardless of its edge.

It enumerates every tool actually registered on the FastMCP server (not a
hand-kept list — it reads `mcp.list_tools()`), invokes each with safe default
arguments, and checks the three things an MCP client depends on:

  CONTRACT 1  NEVER RAISES   — a tool must return an error payload, not throw.
                               A raised exception crashes the JSON-RPC turn.
  CONTRACT 2  JSON-SAFE      — the result must serialize over the transport.
  CONTRACT 3  RETURNS A DICT — every tool here is typed `-> dict[str, Any]`.

Each tool lands in one bucket:
  OK        well-formed dict with real content
  DEGRADED  well-formed dict but signals failure ({"error": ...} / empty) —
            ACCEPTABLE: the tool failed gracefully (e.g. data source down)
  CRASH     raised an exception            -> CONTRACT 1 violated  (hard fail)
  NONDICT   returned a non-dict            -> CONTRACT 3 violated  (hard fail)
  NONSERIAL result won't JSON-encode       -> CONTRACT 2 violated  (hard fail)
  TIMEOUT   exceeded the per-tool budget   -> reliability concern  (hard fail)

A short determinism check confirms cache-backed tools return stable output.

Writers (log_decision, render_company_briefing, refresh_price_cache) are
skipped by default to avoid side effects; add --include-writers to test them.

    python -m tests.validate_mcp_tools
    python -m tests.validate_mcp_tools --timeout 60 --include-writers
    python -m tests.validate_mcp_tools --only decide,score_stock,get_quote
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import egx_mcp.server as server

ROOT = Path(__file__).parent.parent
_SAMPLE_CSV = ROOT / "sample_portfolio.csv"

# Tools with side effects (writes/network mutation) — skipped unless asked.
_WRITERS = {"log_decision", "render_company_briefing", "refresh_price_cache"}

# Heavy BATCH tools: they fetch across the whole universe / run real backtests,
# so tens of seconds is expected, not a hang. They get a larger budget and, if
# they exceed the normal timeout but finish within it, are flagged SLOW (a
# warning) rather than TIMEOUT (a hard failure). Only a heavy tool that blows
# the heavy budget is treated as genuinely hung.
_HEAVY = {"backtest_agentic", "backtest_strategy", "scan_universe_behavior",
          "scan_short_term_winners", "company_brief_full", "optimize_portfolio",
          "screen_stocks", "weekly_top_picks"}
_HEAVY_MULT = 4  # heavy budget = timeout * this (genuine-hang ceiling)

# Safe values for REQUIRED parameters, keyed by parameter name. Optional params
# are left at their declared defaults.
_ARGS = {
    "ticker": "COMI",
    "tickers": ["COMI", "SWDY", "HRHO"],
    "name": "EGX30",
    "verdict": "BUY",
    "proposed_verdict": "BUY",
    "portfolio_value_egp": 500_000,
    "intended_shares": 1000,
    "csv_path": str(_SAMPLE_CSV) if _SAMPLE_CSV.exists() else None,
}

# Cache/compute tools that should be deterministic across back-to-back calls
# (ignoring volatile timestamp fields). Used for the determinism check.
_DETERMINISTIC = ["price_cache_status", "list_egx_stocks", "get_egp_risk_free_rate"]
_VOLATILE_KEYS = {"timestamp", "asof", "generated_at", "as_of", "now"}


def _build_kwargs(tool) -> dict:
    schema = tool.inputSchema or {}
    required = schema.get("required", [])
    kwargs = {}
    for name in required:
        if name in _ARGS and _ARGS[name] is not None:
            kwargs[name] = _ARGS[name]
        else:
            # Required param we don't have a safe value for -> let it surface.
            kwargs[name] = "COMI" if "ticker" in name else 1
    return kwargs


def _classify(result) -> tuple[str, str]:
    """Return (bucket, note) for a successfully-returned (non-raised) result."""
    if not isinstance(result, dict):
        return "NONDICT", f"returned {type(result).__name__}"
    try:
        json.dumps(result, default=str)
        # default=str would mask non-serializable types; check strict too.
        json.dumps(result)
    except (TypeError, ValueError) as e:
        return "NONSERIAL", str(e)[:60]
    err = result.get("error")
    if err:
        return "DEGRADED", f"error: {str(err)[:50]}"
    if not result or all(v in (None, [], {}, "") for v in result.values()):
        return "DEGRADED", "empty payload"
    return "OK", f"{len(result)} keys"


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def _run_one(fn, kwargs, timeout):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, **kwargs)
        try:
            return ("ok", fut.result(timeout=timeout))
        except FutureTimeout:
            return ("timeout", None)
        except Exception as e:  # noqa: BLE001 — the whole point is to catch raises
            return ("raise", f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=90.0, help="per-tool seconds")
    ap.add_argument("--include-writers", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated tool names")
    args = ap.parse_args()

    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    names = sorted(tools)
    if only:
        names = [n for n in names if n in only]

    print("=" * 72)
    print("EGX MCP — TOOL TRANSPORT RELIABILITY VALIDATION")
    print("=" * 72)
    print(f"registered tools: {len(tools)}   testing: {len(names)}   "
          f"timeout: {args.timeout:.0f}s/tool")

    buckets: dict[str, list[str]] = {k: [] for k in
                                     ("OK", "DEGRADED", "SLOW", "CRASH", "NONDICT",
                                      "NONSERIAL", "TIMEOUT", "SKIP")}
    badge = {"OK": "[ OK ]", "DEGRADED": "[deg ]", "SLOW": "[slow]",
             "CRASH": "[CRSH]", "NONDICT": "[!dic]", "NONSERIAL": "[!ser]",
             "TIMEOUT": "[time]", "SKIP": "[skip]"}
    print()
    for name in names:
        tool = tools[name]
        if name in _WRITERS and not args.include_writers:
            buckets["SKIP"].append(name)
            print(f"  {badge['SKIP']} {name:30} side-effecting; --include-writers to test")
            continue
        fn = getattr(server, name, None)
        if not callable(fn):
            buckets["CRASH"].append(name)
            print(f"  {badge['CRASH']} {name:30} not callable on server module")
            continue
        kwargs = _build_kwargs(tool)
        heavy = name in _HEAVY
        budget = args.timeout * _HEAVY_MULT if heavy else args.timeout
        t0 = time.perf_counter()
        status, payload = _run_one(fn, kwargs, budget)
        dt = time.perf_counter() - t0
        if status == "timeout":
            # Even the heavy budget blown -> genuinely hung -> hard failure.
            bucket, note = "TIMEOUT", f">{budget:.0f}s (hung)"
        elif status == "raise":
            bucket, note = "CRASH", payload
        else:
            bucket, note = _classify(payload)
            # A heavy batch tool that completed but ran long is SLOW, not broken.
            if bucket == "OK" and dt > args.timeout:
                bucket, note = "SLOW", f"{note} — heavy batch, {dt:.0f}s"
        buckets[bucket].append(name)
        print(f"  {badge[bucket]} {name:30} {dt:5.1f}s  {note}")

    # --- Determinism check ---------------------------------------------------
    print("\n" + "-" * 72)
    print("DETERMINISM  (cache-backed tools must be stable back-to-back)")
    print("-" * 72)
    det_fail = []
    for name in _DETERMINISTIC:
        fn = getattr(server, name, None)
        if name not in tools or not callable(fn):
            continue
        try:
            a = _strip_volatile(fn())
            b = _strip_volatile(fn())
            ok = a == b
            print(f"  {'[ OK ]' if ok else '[FAIL]'} {name:30} "
                  f"{'stable' if ok else 'DIFFERS across calls'}")
            if not ok:
                det_fail.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"  [CRSH] {name:30} raised: {type(e).__name__}: {e}")
            det_fail.append(name)

    # --- Summary & verdict ---------------------------------------------------
    hard = (buckets["CRASH"] + buckets["NONDICT"] + buckets["NONSERIAL"]
            + buckets["TIMEOUT"] + det_fail)
    print("\n" + "=" * 72)
    print("SUMMARY")
    for k in ("OK", "DEGRADED", "SLOW", "CRASH", "NONDICT", "NONSERIAL",
              "TIMEOUT", "SKIP"):
        if buckets[k]:
            print(f"  {k:10} {len(buckets[k]):3}   {', '.join(buckets[k])[:200]}")
    print("=" * 72)
    if hard:
        print(f"VERDICT: UNRELIABLE — {len(hard)} contract violation(s).")
        print("  A tool that raises, returns a non-dict/non-serializable payload, or")
        print("  hangs will break the MCP turn in Claude. Fix before relying on it.")
        return 2
    notes = []
    if buckets["SLOW"]:
        notes.append(f"{len(buckets['SLOW'])} heavy batch tool(s) ran long "
                     f"(>{args.timeout:.0f}s) but returned valid output — expected for "
                     "universe scans / backtests; call them sparingly, not inline.")
    if buckets["DEGRADED"]:
        notes.append(f"{len(buckets['DEGRADED'])} tool(s) degraded gracefully (data "
                     "source/network, not an MCP bug) — check if you expected live data.")
    if notes:
        print("VERDICT: CONTRACTS HOLD — transport is reliable. Notes:")
        for n in notes:
            for line in (n[i:i + 68] for i in range(0, len(n), 68)):
                print("  " + line)
        return 0
    print("VERDICT: ALL TOOLS HEALTHY — every tested tool honoured its contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
