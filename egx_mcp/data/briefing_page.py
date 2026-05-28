"""Self-contained HTML company briefing — chart + drivers + fundamentals + verdict.

Renders a single offline .html file for one EGX name: a TradingView
Lightweight Charts candlestick + volume chart drawn from the cached OHLC
(see `price_cache`), alongside the driver decomposition, fundamentals, and
the monthly verdict. The charting library is vendored locally
(`lib/lightweight-charts.standalone.production.js`) and inlined into the
page, so the file opens in any browser with no network.

The chart and drivers come from the offline price cache, so the page
renders even when the live feed is down. The verdict / richer fundamentals
are pulled from `company_brief` when reachable and simply omitted otherwise.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from . import behavior, company_brief, price_cache, project_impact
from .universe import resolve_ticker

log = logging.getLogger("egx-mcp.briefing_page")

_LIB_PATH = Path(__file__).parent / "lib" / "lightweight-charts.standalone.production.js"
_OUT_DIR = Path(__file__).parent / "briefings"


def _fmt(v: Any, suffix: str = "", dash: str = "—") -> str:
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:,.2f}{suffix}"
    return f"{html.escape(str(v))}{suffix}"


def _signed(v: Any, suffix: str = "%") -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}{suffix}"


def _rows(pairs: list[tuple[str, str]]) -> str:
    return "".join(
        f'<div class="row"><span class="k">{html.escape(k)}</span>'
        f'<span class="v">{v}</span></div>'
        for k, v in pairs
    )


def _chart_data(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split cached OHLC rows into Lightweight Charts candle + volume series."""
    candles, vols = [], []
    for r in rows:
        candles.append({
            "time": r["date"],
            "open": r["open"], "high": r["high"],
            "low": r["low"], "close": r["close"],
        })
        up = r["close"] >= r["open"]
        vols.append({
            "time": r["date"],
            "value": r.get("volume") or 0,
            "color": "rgba(38,166,154,0.5)" if up else "rgba(239,83,80,0.5)",
        })
    return candles, vols


def _drivers_panel(prof: dict[str, Any]) -> str:
    d = prof.get("drivers", {})
    risk = prof.get("risk", {})
    betas = d.get("factor_betas", {})
    beta_rows = _rows([
        ("Market (EGX30)", _fmt(betas.get("egx30"))),
        ("USD/EGP", _fmt(betas.get("egp"))),
        ("Brent oil", _fmt(betas.get("brent"))),
        ("Gold", _fmt(betas.get("gold"))),
        ("EM equities", _fmt(betas.get("em"))),
    ])
    stats = _rows([
        ("Systematic R²", _fmt((d.get("systematic_r2") or 0) * 100, "%") if d.get("systematic_r2") is not None else "—"),
        ("Idiosyncratic", _fmt(d.get("idiosyncratic_pct"), "%")),
        ("Daily alpha", _signed(d.get("alpha_daily_pct"))),
        ("Ann. volatility", _fmt(risk.get("annualized_volatility_pct"), "%")),
        ("Max drawdown", _fmt(risk.get("max_drawdown_pct"), "%")),
        ("Trailing return", _signed(risk.get("trailing_return_pct"))),
        ("Momentum 20d", _signed(risk.get("momentum_20d_pct"))),
        ("Momentum 60d", _signed(risk.get("momentum_60d_pct"))),
    ])
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in prof.get("interpretation", []))
    return f"""
    <div class="panel">
      <h2>Factor betas</h2>{beta_rows}
    </div>
    <div class="panel">
      <h2>Risk &amp; driver share</h2>{stats}
    </div>
    <div class="panel wide">
      <h2>What moves this stock</h2>
      <ul class="notes">{notes or '<li>No interpretation available.</li>'}</ul>
    </div>"""


def _offline_signal(prof: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """A transparent technical read from the cached drivers + price trend.

    Not a fundamental verdict — a momentum/trend/risk score derived purely
    from offline data, so the briefing always carries an actionable take.
    """
    d = prof.get("drivers", {})
    risk = prof.get("risk", {})
    closes = [r["close"] for r in rows]
    last = closes[-1]
    sma20 = sum(closes[-20:]) / len(closes[-20:]) if len(closes) >= 20 else None
    sma50 = sum(closes[-50:]) / len(closes[-50:]) if len(closes) >= 50 else None
    peak = max(closes)
    trough = min(closes)
    pct_off_high = (last / peak - 1) * 100 if peak else 0.0
    range_pos = (last - trough) / (peak - trough) * 100 if peak > trough else None

    m20 = risk.get("momentum_20d_pct")
    m60 = risk.get("momentum_60d_pct")
    alpha = d.get("alpha_daily_pct")
    vol = risk.get("annualized_volatility_pct")

    score = 0.0
    reasons: list[str] = []
    if m20 is not None:
        if m20 > 3:
            score += 1; reasons.append(f"20-day momentum positive ({m20:+.1f}%).")
        elif m20 < -3:
            score -= 1; reasons.append(f"20-day momentum negative ({m20:+.1f}%).")
    if m60 is not None:
        if m60 > 5:
            score += 1; reasons.append(f"60-day momentum strong ({m60:+.1f}%).")
        elif m60 < -5:
            score -= 1; reasons.append(f"60-day momentum weak ({m60:+.1f}%).")
    if sma50 is not None:
        if last > sma50:
            score += 1; reasons.append(f"Trading above its 50-day average ({last:,.2f} vs {sma50:,.2f}).")
        else:
            score -= 1; reasons.append(f"Trading below its 50-day average ({last:,.2f} vs {sma50:,.2f}).")
    if pct_off_high > -5:
        score += 1; reasons.append(f"Within {pct_off_high:.1f}% of its period high — uptrend intact.")
    elif pct_off_high < -20:
        score -= 1; reasons.append(f"{pct_off_high:.1f}% off its period high — deep drawdown.")
    if alpha is not None:
        if alpha > 0:
            score += 0.5; reasons.append(f"Positive daily alpha ({alpha:+.3f}%) — outperforming its factor model.")
        elif alpha < 0:
            score -= 0.5; reasons.append(f"Negative daily alpha ({alpha:+.3f}%) — lagging its factor model.")

    if score >= 2:
        signal, cls = "Constructive", "buy"
    elif score <= -2:
        signal, cls = "Cautious", "sell"
    else:
        signal, cls = "Neutral", "hold"

    vb = behavior._vol_bucket(vol)
    risk_note = f"{vb.capitalize()} volatility ({vol:.0f}% annualized) — size accordingly." if vb else None

    return {
        "signal": signal, "cls": cls, "score": round(score, 1),
        "reasons": reasons, "risk_note": risk_note,
        "range_position_pct": round(range_pos, 1) if range_pos is not None else None,
        "pct_off_high": round(pct_off_high, 1),
    }


def _signal_panel(sig: dict[str, Any]) -> str:
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in sig["reasons"])
    rows = _rows([
        ("Signal score", _fmt(sig["score"])),
        ("Range position", _fmt(sig["range_position_pct"], "%")),
        ("Off period high", _signed(sig["pct_off_high"])),
    ])
    risk = f'<div class="muted" style="margin-top:8px">{html.escape(sig["risk_note"])}</div>' if sig.get("risk_note") else ""
    return f"""
    <div class="panel wide">
      <h2>Quantitative signal <span class="badge {sig['cls']}">{html.escape(sig['signal'])}</span>
        <span class="sub" style="font-size:11px; text-transform:none; letter-spacing:0">offline · technical read from cached drivers</span></h2>
      {rows}
      <ul class="notes">{reasons or '<li>Insufficient history for a signal.</li>'}</ul>
      {risk}
    </div>"""


def _fundamentals_panel(prof: dict[str, Any], brief: dict[str, Any] | None) -> str:
    f = prof.get("fundamentals", {})
    if "error" in f and brief:
        f = brief.get("fundamentals", {}) or f
    pairs = [
        ("P/E", _fmt(f.get("pe_ratio"))),
        ("P/B", _fmt(f.get("pb_ratio"))),
        ("ROE", _fmt(f.get("roe_pct"), "%")),
        ("Debt/Equity", _fmt(f.get("debt_to_equity"))),
        ("Dividend yield", _fmt(f.get("dividend_yield_pct"), "%")),
    ]
    if all(v == "—" for _, v in pairs):
        body = '<div class="muted">Fundamentals unavailable offline.</div>'
    else:
        body = _rows(pairs)
    return f'<div class="panel"><h2>Fundamentals</h2>{body}</div>'


def _verdict_panel(brief: dict[str, Any] | None) -> str:
    dec = (brief or {}).get("monthly_decision_v8b", {}) or {}
    verdict = dec.get("verdict")
    if not verdict:
        return """
    <div class="panel wide">
      <h2>Monthly verdict (V8b) <span class="badge hold">N/A</span></h2>
      <div class="muted">The fundamental BUY/HOLD/SELL verdict needs the live feed
      (fundamentals, targets, news). Re-run this briefing online to populate it.</div>
    </div>"""
    cls = {"BUY": "buy", "SELL": "sell"}.get(str(verdict).upper(), "hold")
    drivers = "".join(f"<li>{html.escape(str(x))}</li>" for x in (dec.get("key_drivers") or []))
    risks = "".join(f"<li>{html.escape(str(x))}</li>" for x in (dec.get("key_risks") or []))
    rows = _rows([
        ("Conviction", _fmt(dec.get("conviction"))),
        ("Composite score", _fmt(dec.get("composite_score"))),
        ("Fair value", _fmt(dec.get("fair_value"))),
        ("Upside", _signed(dec.get("upside_pct"))),
    ])
    return f"""
    <div class="panel wide">
      <h2>Monthly verdict <span class="badge {cls}">{html.escape(str(verdict))}</span></h2>
      {rows}
      <div class="cols">
        <div><h3>Key drivers</h3><ul class="notes">{drivers or '<li>—</li>'}</ul></div>
        <div><h3>Key risks</h3><ul class="notes">{risks or '<li>—</li>'}</ul></div>
      </div>
    </div>"""


def _catalyst_panel(impact: dict[str, Any] | None) -> str:
    """Project / catalyst event-study panel: reaction profile + scored news."""
    if not impact or "error" in impact:
        why = (impact or {}).get("error", "needs the cached price history")
        return f"""
    <div class="panel wide">
      <h2>Project &amp; catalyst impact <span class="badge hold">N/A</span></h2>
      <div class="muted">Event-study reaction profile unavailable — {html.escape(str(why))}.</div>
    </div>"""

    prof = impact.get("reaction_profile", {})
    scan = impact.get("catalyst_scan", {})
    band = prof.get("catalyst_band_pct") or [None, None]
    net = scan.get("net_expected_move_pct")
    tone = scan.get("net_tone", "mixed/neutral")
    tone_cls = "buy" if tone == "bullish" else "sell" if tone == "bearish" else "hold"

    stats = _rows([
        ("Typical catalyst move", _fmt(prof.get("typical_catalyst_move_pct"), "%")),
        ("Low / high band", f"{_fmt(band[0], '%')} / {_fmt(band[1], '%')}" if band[0] is not None else "—"),
        ("Stock-specific share", _fmt(prof.get("idiosyncratic_pct"), "%")),
        ("Event days (2σ)", _fmt(prof.get("n_event_days_2sigma"))),
        ("Observations", _fmt(prof.get("n_obs"))),
    ])

    cats = scan.get("catalysts") or []
    if cats:
        items = "".join(
            f'<li><span class="badge {("buy" if c["tone"]=="positive" else "sell" if c["tone"]=="negative" else "hold")}">'
            f'{_signed(c.get("est_impact_pct"))}</span> '
            f'{html.escape((c.get("date") or "")[:10])} · {html.escape(str(c.get("source") or ""))} — '
            f'{html.escape(str(c.get("title") or ""))}</li>'
            for c in cats[:8]
        )
        news = (
            f'<h3>Recent project headlines '
            f'<span class="badge {tone_cls}">net {html.escape(tone)} {_signed(net)}</span></h3>'
            f'<ul class="notes">{items}</ul>'
        )
    elif not scan.get("news_available"):
        news = '<div class="muted" style="margin-top:8px">No live news available (offline) — showing reaction profile only.</div>'
    else:
        news = '<div class="muted" style="margin-top:8px">No project/catalyst headlines in the recent news flow.</div>'

    notes = "".join(f"<li>{html.escape(n)}</li>" for n in impact.get("interpretation", []))
    return f"""
    <div class="panel wide">
      <h2>Project &amp; catalyst impact
        <span class="sub" style="font-size:11px; text-transform:none; letter-spacing:0">event study · abnormal returns from cached factor model</span></h2>
      {stats}
      <ul class="notes" style="margin-top:10px">{notes}</ul>
      {news}
    </div>"""


def _html_page(
    canonical: str, name: str, sector: str,
    quote: dict[str, Any], candles: list[dict], vols: list[dict],
    panels: str, generated: str,
) -> str:
    lib_js = _LIB_PATH.read_text(encoding="utf-8")
    change = quote.get("change_pct")
    change_cls = "up" if (change or 0) >= 0 else "down"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(canonical)} — {html.escape(name)} briefing</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --line:#21262d; --fg:#c9d1d9; --mut:#8b949e; --up:#26a69a; --down:#ef5350; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:20px 28px; border-bottom:1px solid var(--line); }}
  h1 {{ margin:0; font-size:22px; }}
  .sub {{ color:var(--mut); margin-top:4px; }}
  .price {{ font-size:28px; font-weight:600; }}
  .up {{ color:var(--up); }} .down {{ color:var(--down); }}
  #chart {{ height:420px; margin:18px 28px; border:1px solid var(--line); border-radius:8px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; padding:0 28px 28px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px 18px; }}
  .panel.wide {{ grid-column:1 / -1; }}
  h2 {{ font-size:14px; margin:0 0 12px; color:var(--fg); text-transform:uppercase; letter-spacing:.04em; }}
  h3 {{ font-size:12px; color:var(--mut); margin:8px 0 4px; text-transform:uppercase; }}
  .row {{ display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed var(--line); }}
  .row:last-child {{ border-bottom:0; }}
  .k {{ color:var(--mut); }} .v {{ font-variant-numeric:tabular-nums; }}
  .notes {{ margin:0; padding-left:18px; }} .notes li {{ margin:4px 0; }}
  .muted {{ color:var(--mut); font-style:italic; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:8px; }}
  .badge {{ font-size:12px; padding:2px 10px; border-radius:12px; vertical-align:middle; }}
  .badge.buy {{ background:rgba(38,166,154,.2); color:var(--up); }}
  .badge.sell {{ background:rgba(239,83,80,.2); color:var(--down); }}
  .badge.hold {{ background:rgba(139,148,158,.2); color:var(--mut); }}
  footer {{ color:var(--mut); padding:0 28px 24px; font-size:12px; }}
</style></head>
<body>
<header>
  <h1>{html.escape(canonical)} · {html.escape(name)}</h1>
  <div class="sub">{html.escape(sector)} &nbsp;·&nbsp; as of {html.escape(quote.get('date') or '—')}</div>
  <div style="margin-top:10px">
    <span class="price">{_fmt(quote.get('price'))}</span>
    <span class="{change_cls}" style="margin-left:10px">{_signed(change)}</span>
    <span class="sub" style="margin-left:14px">vol {_fmt(quote.get('volume'))}</span>
  </div>
</header>
<div id="chart"></div>
<div class="grid">{panels}</div>
<footer>Generated {html.escape(generated)} · prices &amp; drivers from price_cache (investing.com) · chart by TradingView Lightweight Charts</footer>
<script>{lib_js}</script>
<script>
  const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
    layout: {{ background: {{ color:'#0d1117' }}, textColor:'#c9d1d9' }},
    grid: {{ vertLines:{{ color:'#21262d' }}, horzLines:{{ color:'#21262d' }} }},
    rightPriceScale: {{ borderColor:'#21262d' }},
    timeScale: {{ borderColor:'#21262d' }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
  }});
  const candle = chart.addCandlestickSeries({{
    upColor:'#26a69a', downColor:'#ef5350', borderVisible:false,
    wickUpColor:'#26a69a', wickDownColor:'#ef5350',
  }});
  candle.setData({json.dumps(candles)});
  const vol = chart.addHistogramSeries({{
    priceFormat:{{ type:'volume' }}, priceScaleId:'',
  }});
  vol.priceScale().applyOptions({{ scaleMargins:{{ top:0.82, bottom:0 }} }});
  vol.setData({json.dumps(vols)});
  chart.timeScale().fitContent();
  new ResizeObserver(es => chart.applyOptions({{ width: es[0].contentRect.width }}))
    .observe(document.getElementById('chart'));
</script>
</body></html>"""


def render(ticker: str, out_dir: str | None = None) -> dict[str, Any]:
    """Render a self-contained HTML briefing for one EGX name; return its path."""
    canonical, _yahoo, name = resolve_ticker(ticker)

    rows = price_cache.get_prices(canonical)
    if not rows:
        return {
            "ticker": canonical,
            "error": "no cached prices — run refresh_price_cache first",
        }

    quote = price_cache.get_quote(canonical)
    prof = behavior.stock_behavior(canonical)
    name = prof.get("name") or name
    sector = prof.get("sector") or "—"

    brief = None
    try:
        brief = company_brief.brief(canonical)
    except Exception as e:
        log.warning(f"company_brief unavailable for {canonical}: {e}")

    impact = None
    try:
        impact = project_impact.project_impact(canonical)
    except Exception as e:
        log.warning(f"project_impact unavailable for {canonical}: {e}")

    sig = _offline_signal(prof, rows)
    candles, vols = _chart_data(rows)
    panels = (
        _drivers_panel(prof)
        + _signal_panel(sig)
        + _fundamentals_panel(prof, brief)
        + _catalyst_panel(impact)
        + _verdict_panel(brief)
    )
    generated = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    page = _html_page(canonical, name, sector, quote, candles, vols, panels, generated)

    target = Path(out_dir) if out_dir else _OUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{canonical}_briefing.html"
    path.write_text(page, encoding="utf-8")

    return {
        "status": "ok",
        "ticker": canonical,
        "name": name,
        "path": str(path),
        "bars": len(rows),
        "date_range": {"start": rows[0]["date"], "end": rows[-1]["date"]},
        "signal": sig["signal"],
        "signal_score": sig["score"],
        "has_verdict": bool(brief and (brief.get("monthly_decision_v8b") or {}).get("verdict")),
    }
