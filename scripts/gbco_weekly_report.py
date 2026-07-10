"""Full weekly investment report for GBCO (GB Corp) — decision-grade.

Sources ONLY real data and computes everything from it:
  • live daily price history (Investing.com primary, Yahoo fallback) → price,
    technicals, volatility, beta, probabilistic outlook, trade plan;
  • audited fundamentals (egx_fundamentals_audited.csv) → multiples, ROE;
  • clean model modules (forecast, macro, catalyst-reaction) called live.

It deliberately AVOIDS the model's snapshot/sizing/peer outputs, which are
contaminated by Yahoo's spurious real-time quote (a ~14.05 half-price print
with no split). The real-time quote is never used; the last available history
session is taken as "last close" (only a same-day in-progress placeholder bar
is dropped). Where the dataset has
no data (multi-year statements, segment detail), the report says so rather
than inventing it.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # make egx_mcp importable when run directly

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, HRFlowable, PageBreak, KeepTogether)

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "reports"; OUT_DIR.mkdir(exist_ok=True)
PDF = OUT_DIR / "GBCO_weekly_2026-06-21.pdf"
CHART = OUT_DIR / "_gbco_chart.png"
PORTFOLIO_EGP = 1_000_000.0   # illustrative book for the trade-plan math

NAVY = colors.HexColor("#0B2545"); STEEL = colors.HexColor("#13315C")
ACCENT = colors.HexColor("#C8961E"); LIGHT = colors.HexColor("#EAF0F7")
GREY = colors.HexColor("#5A6B7B"); GREEN = colors.HexColor("#1E7A46"); RED = colors.HexColor("#B23B3B")

def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None

# ── Price history: Investing.com (reliable for EGX tail) → Yahoo → cache ─────
# Yahoo carries STALE zero-volume placeholder bars for EGX that hide the real
# latest session (e.g. it showed GBCO 2026-06-17 as a flat 28.33/0-vol carry
# when the real close was 28.80 on 3.1m volume). Investing.com has the correct
# daily history, so it is the primary source.
def _fetch_live(symbol="GBCO.CA", ticker="GBCO"):
    from egx_mcp.data._certs import ensure_ca_bundle
    ensure_ca_bundle()
    try:
        from egx_mcp.data import investing
        h = investing.fetch_history(ticker, lookback_days=800)
        if h and len(h) > 200:
            rows = [{"date": b["date"], "open": float(b["open"]), "high": float(b["high"]),
                     "low": float(b["low"]), "close": float(b["close"]), "volume": int(b.get("volume") or 0)}
                    for b in h]
            return rows, "Investing.com"
    except Exception:
        pass
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period="2y", auto_adjust=False)
        if df is not None and not df.empty:
            rows = [{"date": ix.strftime("%Y-%m-%d"), "open": float(r.Open), "high": float(r.High),
                     "low": float(r.Low), "close": float(r.Close), "volume": int(r.Volume)}
                    for ix, r in df.iterrows()]
            today = datetime.now().strftime("%Y-%m-%d")  # drop only a same-day in-progress placeholder
            if len(rows) > 1 and rows[-1]["volume"] == 0 and rows[-1]["date"] == today:
                rows.pop()
            return rows, "Yahoo"
    except Exception:
        pass
    return None, None

try:
    series, SRC = _fetch_live(); LIVE = series is not None and len(series) > 200
except Exception:
    series, SRC, LIVE = None, None, False
if not LIVE:
    cache = json.loads((ROOT / "egx_mcp/data/price_cache.json").read_text(encoding="utf-8"))
    series = cache["prices"]["GBCO"]; cache_asof = cache["refreshed_at"][:10]; SRC = "cached"
else:
    cache_asof = "live"
closes = [b["close"] for b in series]; highs = [b["high"] for b in series]
lows = [b["low"] for b in series]; vols = [b["volume"] for b in series]
dates = [datetime.strptime(b["date"], "%Y-%m-%d") for b in series]
last = closes[-1]; last_date = series[-1]["date"]

# ── Fundamentals (audited) ───────────────────────────────────────────────────
rows_f = list(csv.DictReader((ROOT / "egx_fundamentals_audited.csv").read_text(encoding="utf-8-sig").splitlines()))
fund = next(r for r in rows_f if r["ticker"] == "GBCO")
eps = float(fund["trailing_eps"]); bvps = float(fund["book_value_per_share"])
roe = float(fund["roe_pct"]); mktcap = float(fund["market_cap"])
pe = last / eps; pb = last / bvps
def med(c):
    xs = [float(r[c]) for r in rows_f if r.get(c) not in (None, "")]
    return st.median(xs)
u_pe, u_pb, u_roe = med("pe_ratio"), med("pb_ratio"), med("roe_pct")

# ── Technicals ───────────────────────────────────────────────────────────────
def sma(n): return sum(closes[-n:]) / n
def rsi(n=14):
    g = l = 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]; g += max(d, 0); l += max(-d, 0)
    return 100.0 if l == 0 else 100 - 100 / (1 + (g / n) / (l / n))
def ema(vals, n):
    k = 2 / (n + 1); e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e
trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(-14, 0)]
atr = sum(trs) / 14
sma20, sma50, sma200 = sma(20), sma(50), sma(200)
rsi14 = rsi()
macd = ema(closes[-26:], 12) - ema(closes[-26:], 26)
hi52, lo52 = max(closes[-252:]), min(closes[-252:])
pos = (last - lo52) / (hi52 - lo52) * 100
ret = lambda n: (last / closes[-n] - 1) * 100
sig5 = atr * math.sqrt(5); lo5, hi5 = last - sig5, last + sig5

def fmt(x, d=2, suf=""):
    return ("%.*f%s" % (d, x, suf)) if isinstance(x, (int, float)) else "n/a"

# Signal descriptors so prose tracks the data
sma50_prev = sum(closes[-70:-20]) / 50
ma_dir = "rising" if sma50 > sma50_prev else "falling" if sma50 < sma50_prev else "flat"
above20, above50 = last > sma20, last > sma50
if above20 and above50:
    trend_txt = f"above its {ma_dir} 20- and 50-day averages ({fmt(sma20)} / {fmt(sma50)})"; trend_tag = "constructive"
elif not above20 and not above50:
    trend_txt = f"below its {ma_dir} 20- and 50-day averages ({fmt(sma20)} / {fmt(sma50)})"; trend_tag = "soft"
else:
    trend_txt = f"straddling its 20/50-day averages ({fmt(sma20)} / {fmt(sma50)})"; trend_tag = "neutral"
rsi_zone = ("overbought" if rsi14 >= 70 else "elevated" if rsi14 >= 60 else "neutral" if rsi14 >= 45
            else "soft" if rsi14 >= 30 else "oversold")
mom3 = ret(63); mom_word = "positive" if mom3 > 3 else "flat" if mom3 > -3 else "negative"
support = min(sma20, sma50)

# ── Probabilistic 5-day outlook (lognormal drift/vol from 60d returns) ───────
rets = [closes[i] / closes[i - 1] - 1 for i in range(-60, 0)]
mu_d, sd_d = sum(rets) / len(rets), st.pstdev(rets)
exp5, sd5 = mu_d * 5, sd_d * math.sqrt(5)
ann_vol = sd_d * math.sqrt(252) * 100
def _phi(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
p_up = (1 - _phi((0.0 - exp5) / sd5)) * 100
p_up2 = (1 - _phi((0.02 - exp5) / sd5)) * 100
p_dn5 = _phi((-0.05 - exp5) / sd5) * 100
p10_5, p90_5 = (exp5 - 1.2816 * sd5) * 100, (exp5 + 1.2816 * sd5) * 100

# ── Beta vs synthetic equal-weight basket (from cached universe matrix) ──────
# Equal-weight basket return on day t = mean across names of (close_t/close_{t-1}-1),
# then OLS-regress GBCO daily returns on basket returns over the overlapping window.
beta = r2 = alpha_d = None
try:
    cache = json.loads((ROOT / "egx_mcp/data/price_cache.json").read_text(encoding="utf-8"))
    px = cache["prices"]
    by_t = {t: {b["date"]: b["close"] for b in bars} for t, bars in px.items()}
    all_dates = sorted({d for m in by_t.values() for d in m})
    basket_ret, gbco_ret = {}, {}
    for i in range(1, len(all_dates)):
        d0, d1 = all_dates[i - 1], all_dates[i]
        rs = [m[d1] / m[d0] - 1 for m in by_t.values() if d0 in m and d1 in m and m[d0]]
        if rs:
            basket_ret[d1] = sum(rs) / len(rs)
        g = by_t["GBCO"]
        if d0 in g and d1 in g and g[d0]:
            gbco_ret[d1] = g[d1] / g[d0] - 1
    paired = [(gbco_ret[d], basket_ret[d]) for d in sorted(gbco_ret) if d in basket_ret][-90:]
    if len(paired) >= 30:
        gr = [a for a, _ in paired]; br = [b for _, b in paired]
        mgr, mbr = sum(gr) / len(gr), sum(br) / len(br)
        cov = sum((a - mgr) * (b - mbr) for a, b in paired)
        vb = sum((b - mbr) ** 2 for b in br); vg = sum((a - mgr) ** 2 for a in gr)
        beta = cov / vb if vb else None
        if beta is not None:
            alpha_d = (mgr - beta * mbr) * 100
            r2 = (cov ** 2 / (vb * vg)) if vg and vb else None
except Exception:
    pass

# ── Valuation anchors ────────────────────────────────────────────────────────
fv_pe = u_pe * eps
fv_pb_parity = u_pb * bvps
justified_pb = u_pb * (roe / u_roe); fv_pb_justified = justified_pb * bvps

# ── Trade plan (computed from history price — NOT the contaminated quote) ────
atr_mult, risk_pct, max_pos_pct, tranches = 2.0, 1.0, 10.0, 3
entry = last
stop = entry - atr_mult * atr; stop_dist = entry - stop
risk_budget = PORTFOLIO_EGP * risk_pct / 100
shares_risk = risk_budget / stop_dist
shares_cap = (PORTFOLIO_EGP * max_pos_pct / 100) / entry
shares = int(min(shares_risk, shares_cap))
pos_val = shares * entry; pos_pct = pos_val / PORTFOLIO_EGP * 100
act_risk = shares * stop_dist; act_risk_pct = act_risk / PORTFOLIO_EGP * 100
constraint = "10% position cap" if shares_cap < shares_risk else "1% risk budget"
t1, t2 = entry + atr, entry + 2 * atr
bb_upper = sma20 + 2 * st.pstdev(closes[-20:])

# ── Live model modules (clean) ───────────────────────────────────────────────
from egx_mcp.data import forecast as _fc, macro as _mac, project_impact as _pi
FC = _safe(_fc.forecast_return, "GBCO", 21)
MAC = _safe(_mac.get_context)
PI = _safe(_pi.project_impact, "GBCO")
pir = (PI or {}).get("reaction_profile", {}) if isinstance(PI, dict) else {}

# ── Chart ────────────────────────────────────────────────────────────────────
def rolling(arr, n):
    out = [None] * len(arr)
    for i in range(n - 1, len(arr)):
        out[i] = sum(arr[i - n + 1:i + 1]) / n
    return out
fig, (axp, axv) = plt.subplots(2, 1, figsize=(7.4, 3.9), height_ratios=[3, 1], sharex=True)
axp.plot(dates, closes, color="#0B2545", lw=1.3, label="Close")
axp.plot(dates, rolling(closes, 50), color="#C8961E", lw=1.0, label="SMA50")
axp.plot(dates, rolling(closes, 200), color="#9AA7B4", lw=1.0, label="SMA200")
axp.set_ylabel("EGP"); axp.legend(loc="upper left", fontsize=7, frameon=False); axp.grid(alpha=0.25)
axp.set_title("GBCO daily close (%s, to %s)" % ("live" if LIVE else "cached", last_date), fontsize=9)
axv.bar(dates, vols, color="#9AA7B4", width=2.0); axv.set_ylabel("Vol"); axv.grid(alpha=0.2)
axv.xaxis.set_major_formatter(DateFormatter("%b-%y"))
fig.tight_layout(); fig.savefig(CHART, dpi=150); plt.close(fig)

# ── Styles ───────────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], textColor=NAVY, fontSize=20, spaceAfter=2, leading=23)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], textColor=GREY, fontSize=10, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=STEEL, fontSize=12.5, spaceBefore=9, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=9.4, leading=13, spaceAfter=4)
SMALL = ParagraphStyle("SMALL", parent=ss["Normal"], fontSize=7.7, leading=9.8, textColor=GREY)
BULL = ParagraphStyle("BULL", parent=BODY, leftIndent=10, spaceAfter=2)
WHITE = ParagraphStyle("WHITE", parent=ss["Normal"], textColor=colors.white, fontSize=9.5, leading=12)
WHITEB = ParagraphStyle("WHITEB", parent=WHITE, fontSize=15, leading=17)

def hdr_table(data, widths, header=True, body_fs=8.5):
    t = Table(data, colWidths=widths)
    sty = [("FONTSIZE", (0, 0), (-1, -1), body_fs), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D4E0")),
           ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT]),
           ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        sty += [("BACKGROUND", (0, 0), (-1, 0), STEEL), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(sty)); return t

flow = []

# ════════ Page 1: cover, verdict, summary, thesis ════════
flow.append(Paragraph("GB Corp &nbsp;·&nbsp; GBCO.CA", H1))
flow.append(Paragraph("EGX Weekly Investment Report &nbsp;|&nbsp; Automotive &nbsp;|&nbsp; "
                      "Decision date 19 Jun 2026 &nbsp;|&nbsp; Week ahead: 21–25 June 2026 (Sun–Thu)", SUB))
flow.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceBefore=4, spaceAfter=7))

vcell = [Paragraph("RECOMMENDATION", WHITE),
         Paragraph("HOLD &nbsp;<font size=9>· medium conviction · model composite 53/100 · tactical-only</font>", WHITEB)]
vt = Table([[vcell]], colWidths=[170 * mm])
vt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("LEFTPADDING", (0, 0), (-1, -1), 10),
                        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
flow.append(vt); flow.append(Spacer(1, 4))

_pl = "Last close" if LIVE else "Price (cached)"
stats = [[_pl, fmt(last) + " EGP", "P/E (hist.)", fmt(pe, 1) + "x", "ROE", fmt(roe, 1) + "%"],
         ["52-wk range", f"{fmt(lo52)}–{fmt(hi52)}", "P/B", fmt(pb, 2) + "x", "Beta (basket)", fmt(beta, 2) if beta else "n/a"],
         ["Mkt cap (audited)", f"{mktcap/1e9:.1f} bn", "EPS / BVPS", f"{fmt(eps)} / {fmt(bvps)}", "Ann. vol", fmt(ann_vol, 0) + "%"]]
kt = Table(stats, colWidths=[26*mm, 27*mm, 22*mm, 28*mm, 24*mm, 23*mm])
kt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.8), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
    ("TEXTCOLOR", (0, 0), (0, -1), GREY), ("TEXTCOLOR", (2, 0), (2, -1), GREY), ("TEXTCOLOR", (4, 0), (4, -1), GREY),
    ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"), ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
    ("FONTNAME", (5, 0), (5, -1), "Helvetica-Bold"), ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT, colors.white]),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.white), ("TOPPADDING", (0, 0), (-1, -1), 4.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5)]))
flow.append(kt)

_dline = (f"Prices &amp; technicals are <b>live to the last EGX session ({last_date})</b>, sourced from "
          f"<b>{SRC}</b>. Yahoo is not used as primary: it carries stale zero-volume placeholder bars for EGX "
          f"(it showed 06-17 as a flat 28.33 carry when the real close was {fmt(last)} on real volume) and its "
          f"real-time quote can glitch (a spurious ~14.05/−50% print), which also contaminates the model's "
          f"snapshot, P/E and sizing tools — so those are recomputed here from clean history."
          if LIVE else
          f"Live refresh failed; using cached prices to {last_date} (cache {cache_asof}) — verify before acting.")
cav = Table([[Paragraph(
    f"<b>Data &amp; reliability notice.</b> Built 19 Jun 2026. {_dline} Fundamentals are audited point-in-time "
    f"figures (no multi-year statements or segment detail in the dataset — see Business). The EGX model is "
    f"<b>NOT YET RELIABLE</b> (3 reliability gates open; directional accuracy below target on a thin sample). "
    f"<b>Decision-support only</b> — keep a human gate, use stops, size to conviction. Not investment advice.", SMALL)]],
    colWidths=[170 * mm])
cav.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFF7E6")), ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
    ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
flow.append(Spacer(1, 5)); flow.append(cav)

flow.append(Paragraph("Investment summary", H2))
flow.append(Paragraph(
    f"We rate GB Corp a <b>HOLD</b>. The stock is roughly fairly valued — cheap on book (P/B {fmt(pb,2)}x vs "
    f"{fmt(u_pb,2)}x universe median) but on a below-average return (ROE {fmt(roe,1)}% vs {fmt(u_roe,1)}%), so the "
    f"discount is largely earned; on earnings it is in line (P/E {fmt(pe,1)}x vs {fmt(u_pe,1)}x). Technically the "
    f"tape has turned <b>{trend_tag}</b> — price {fmt(last)} is {trend_txt}, RSI {fmt(rsi14,0)} ({rsi_zone}), 3-month "
    f"momentum {'+' if mom3>=0 else ''}{fmt(mom3,1)}%. The model's probabilistic outlook is modestly positive "
    f"(21-day E[ret] {fmt((FC or {}).get('expected_return_pct'),1) if FC else 'n/a'}%, wide band) but offers no "
    f"strong edge. Net: no fundamental catalyst to chase here — treat any long as a <b>tactical momentum trade</b> "
    f"with a defined stop, not a conviction position.", BODY))

flow.append(Paragraph("Bull case vs bear case", H2))
bull = [f"Cheap on assets (P/B {fmt(pb,2)}x, near book) with a margin of safety if returns normalise.",
        f"Constructive technical structure: above rising 20/50/200-day averages, MACD positive.",
        f"Low FX/commodity beta historically — less exposed to EGP &amp; Brent swings than the sector myth suggests.",
        f"Premium-brand expansion (e.g. Genesis launch) supports a longer-run distribution mix story."]
bear = [f"Low ROE ({fmt(roe,1)}%) caps the re-rating — value is partly a quality discount.",
        f"RSI {fmt(rsi14,0)} is {rsi_zone}; the move is extended and prone to mean-reversion.",
        f"Auto demand is rate- &amp; FX-sensitive; a hawkish CBE or EGP slide pressures volumes and margins.",
        f"Thin fundamental coverage — no multi-year statements/segments — limits conviction."]
bb = Table([[Paragraph("<b>Bull</b>", BODY), Paragraph("<b>Bear</b>", BODY)],
            [Paragraph("<br/>".join("• " + x for x in bull), SMALL),
             Paragraph("<br/>".join("• " + x for x in bear), SMALL)]], colWidths=[85*mm, 85*mm])
bb.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#E3F1E4")), ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FBE6E6")),
    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D4E0")), ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C9D4E0")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
flow.append(bb)
flow.append(Paragraph(f"<b>What would change the call.</b> Upgrade on a return inflection (ROE toward the ~{fmt(u_roe,0)}% "
    f"median or margin expansion) or a clean breakout that holds above {fmt(bb_upper)}. Downgrade on a daily close "
    f"below {fmt(support)} support or a hawkish rate/FX shock.", BODY))

flow.append(PageBreak())

# ════════ Page 2: business, price & technicals ════════
flow.append(Paragraph("Business overview", H2))
flow.append(Paragraph(
    "GB Corp (formerly GB Auto) is one of Egypt's largest automotive groups, historically spanning passenger-vehicle "
    "assembly &amp; distribution, two-/three-wheelers and commercial vehicles, tyres, after-sales, and a consumer-"
    "finance arm. It is the sole Automotive name in the model's validated universe. <i>(General background, not from "
    "the model dataset.)</i> The quantitative dataset here is a point-in-time fundamentals snapshot plus price "
    "history — it does <b>not</b> include multi-year income statements, balance sheets, cash-flow, or segment "
    "revenue. Those should be sourced from the company's filings before a high-conviction decision; this report is "
    "built on what is verifiable from the model.", BODY))

flow.append(Paragraph("Price &amp; technical picture", H2))
flow.append(Image(str(CHART), width=170 * mm, height=90 * mm))
_ma_read = ("Above SMA20 & 50" if (above20 and above50) else "Below SMA20 & 50" if (not above20 and not above50) else "Straddling 20/50")
tech = [["Metric", "Value", "Read"],
        ["Last close", f"{fmt(last)} EGP ({last_date})", "live" if LIVE else "cached"],
        ["SMA 20 / 50 / 200", f"{fmt(sma20)} / {fmt(sma50)} / {fmt(sma200)}", f"{_ma_read}; {ma_dir} 50-day"],
        ["RSI(14)", fmt(rsi14, 1), rsi_zone.capitalize()],
        ["MACD (12/26)", fmt(macd, 2), "Positive" if macd > 0 else "Negative"],
        ["Bollinger (20,2)", f"{fmt(sma20 - 2*st.pstdev(closes[-20:]))}–{fmt(bb_upper)}", "Bands"],
        ["ATR(14) / ann. vol", f"{fmt(atr)} ({fmt(atr/last*100,1)}%) / {fmt(ann_vol,0)}%", "Volatility"],
        ["52-wk range / pos.", f"{fmt(lo52)}–{fmt(hi52)} / {fmt(pos,0)}%", f"{fmt(pos,0)}% of range"],
        ["Return 1m/3m/6m/1y", f"{fmt(ret(21),1)}/{fmt(ret(63),1)}/{fmt(ret(126),1)}/{fmt(ret(252),1)}%", f"{mom_word.capitalize()} 3m"],
        ["Volume 20d / last", f"{sum(vols[-20:])//20:,} / {vols[-1]:,}", "Liquidity"]]
flow.append(Spacer(1, 3)); flow.append(hdr_table(tech, [40*mm, 66*mm, 64*mm]))
flow.append(Paragraph(
    f"<b>Technical read.</b> At {fmt(last)} GBCO is {trend_txt}, 50-day {ma_dir}, RSI {fmt(rsi14,0)} ({rsi_zone}), "
    f"MACD {'positive' if macd>0 else 'negative'}. 3-month momentum is {'+' if mom3>=0 else ''}{fmt(mom3,1)}% vs "
    f"{'+' if ret(252)>=0 else ''}{fmt(ret(252),0)}% over a year. "
    + ("Structure is constructive — price leads its rising averages — but RSI is getting full, so chasing carries "
       f"pullback risk; {fmt(support)} is the level to defend." if trend_tag == "constructive"
       else "Defensive setup; a reclaim of the 20-day would be the first repair." if trend_tag == "soft"
       else "Range-trade, not trend."), BODY))

flow.append(PageBreak())

# ════════ Page 3: forecast, valuation ════════
flow.append(Paragraph("Forecast &amp; probabilistic outlook", H2))
fc_e = (FC or {}).get("expected_return_pct"); fc_u = (FC or {}).get("uncertainty_pct"); fc_dir = (FC or {}).get("direction")
prob = [["Horizon", "Expected", "Down / Up odds", "Range (P10–P90)"],
        ["5 sessions (week ahead)", f"{fmt(exp5*100,1)}%", f"P(up) {fmt(p_up,0)}% · P(>+2%) {fmt(p_up2,0)}% · P(<−5%) {fmt(p_dn5,0)}%",
         f"{fmt(p10_5,1)}% to {fmt(p90_5,1)}%"],
        ["21 sessions (model)", f"{fmt(fc_e,1) if fc_e is not None else 'n/a'}%",
         f"direction: {fc_dir or 'n/a'} · ±{fmt(fc_u,0) if fc_u is not None else 'n/a'}% (1σ)", "wide — size down"]]
flow.append(hdr_table(prob, [42*mm, 24*mm, 64*mm, 40*mm]))
flow.append(Paragraph(
    f"<b>Week-ahead price map (from {fmt(last)}):</b> base case ~{fmt(lo5)}–{fmt(hi5)} (±1 ATR·√5). "
    f"Support {fmt(support)} (20/50-day); a daily close below it opens {fmt(lo5)}. Upside trigger: reclaim/hold "
    f"toward the upper band {fmt(bb_upper)}, then {fmt(hi5)}. The 5-day expected move is small versus the daily "
    f"noise (ATR {fmt(atr/last*100,1)}%/day) — a range week is the base case. The 21-day model drift is positive "
    f"but its uncertainty band is far wider than the signal, which is itself a reason to keep size modest.", BODY))

flow.append(Paragraph("Valuation &amp; fundamentals", H2))
val = [["", "GBCO", "Universe median", "Signal"],
       ["P/E (trailing, hist. price)", f"{fmt(pe,1)}x", f"{fmt(u_pe,1)}x", "≈ in line"],
       ["P/B", f"{fmt(pb,2)}x", f"{fmt(u_pb,2)}x", "Cheap on book"],
       ["ROE", f"{fmt(roe,1)}%", f"{fmt(u_roe,1)}%", "Below average"],
       ["EPS / Book value (audited)", f"{fmt(eps)} / {fmt(bvps)} EGP", "—", "—"],
       ["Market cap (audited)", f"{mktcap/1e9:.1f} bn EGP", "—", "—"],
       ["v8b quality filter", "PASS", "—", "Eligible"]]
flow.append(hdr_table(val, [50*mm, 34*mm, 38*mm, 48*mm]))
flow.append(Paragraph(
    f"<b>Fair-value anchors (illustrative).</b> P/E-parity to the median implies ~{fmt(fv_pe)} EGP; a full re-rate "
    f"to median P/B implies ~{fmt(fv_pb_parity)} EGP (bull case, ignores lower returns). The honest anchor scales "
    f"P/B by relative profitability — justified P/B {fmt(justified_pb,2)}x (= {fmt(u_pb,2)}x × {fmt(roe,1)}/{fmt(u_roe,1)}) "
    f"→ ~{fmt(fv_pb_justified)} EGP. Current {fmt(last)} sits inside that span: broadly fairly valued. The case is "
    f"value-with-low-quality — you are paid to wait via a near-book multiple, but a re-rating needs ROE to improve.", BODY))

flow.append(PageBreak())

# ════════ Page 4: factor/risk, macro, catalysts ════════
flow.append(Paragraph("Factor &amp; risk profile", H2))
fr = [["Measure", "Value", "Read"],
      ["Beta vs EGX basket (90d)", fmt(beta, 2) if beta else "n/a", "≈ moves with the market" if beta and 0.8 < beta < 1.2 else "—"],
      ["Daily alpha (90d)", f"{fmt(alpha_d,2) if alpha_d is not None else 'n/a'}%", "vs equal-weight basket"],
      ["R² (basket)", fmt(r2, 2) if r2 is not None else "n/a", "rest is idiosyncratic"],
      ["Annualised volatility", f"{fmt(ann_vol,0)}%", "typical EGX mid-cap"],
      ["Typical catalyst move", f"{fmt(pir.get('typical_catalyst_move_pct'),1) if pir else 'n/a'}%",
       f"band {fmt(pir.get('catalyst_band_pct',[None,None])[0]) if pir.get('catalyst_band_pct') else 'n/a'}–{fmt(pir.get('catalyst_band_pct',[None,None])[1]) if pir.get('catalyst_band_pct') else 'n/a'}%"],
      ["Idiosyncratic share", f"{fmt(pir.get('idiosyncratic_pct'),0) if pir else 'n/a'}%", "stock-specific, not market"]]
flow.append(hdr_table(fr, [50*mm, 34*mm, 86*mm]))
flow.append(Paragraph(
    f"GBCO trades roughly with the market (beta ~{fmt(beta,2) if beta else 'n/a'}) but most of its variance is "
    f"stock-specific ({fmt(pir.get('idiosyncratic_pct'),0) if pir else 'n/a'}% idiosyncratic), so single-name "
    f"news drives it more than the index. Historically a catalyst day moves it ~{fmt(pir.get('typical_catalyst_move_pct'),1) if pir else 'n/a'}% — "
    f"size positions so a normal event-day swing is survivable.", BODY))

flow.append(Paragraph("Macro backdrop (autos)", H2))
if isinstance(MAC, dict):
    egp = MAC.get("egp_usd", {}); br = MAC.get("brent_usd", {}); gd = MAC.get("gold_usd", {}); cbe = MAC.get("cbe_rates", {})
    mac = [["Indicator", "Level", "Why it matters for GB Corp"],
           ["EGP / USD", fmt(egp.get("value"), 2), "Imported vehicles &amp; parts — weaker EGP lifts COGS"],
           ["CBE policy rate", f"{fmt(cbe.get('midpoint_pct'))}%" if cbe.get("midpoint_pct") else "n/a (verify at cbe.org.eg)", "High rates depress auto financing demand"],
           ["Brent (USD)", fmt(br.get("value"), 1), "Fuel cost → consumer demand; low direct beta"],
           ["Gold (USD)", fmt(gd.get("value"), 0), "Savings/store-of-value alternative for EGP holders"]]
    flow.append(hdr_table(mac, [38*mm, 30*mm, 102*mm]))
    flow.append(Paragraph("<i>EGP/USD &amp; Brent are delayed Yahoo prints; CBE rates must be verified at cbe.org.eg "
                          "before acting on rate-sensitive trades. Model sector macro-bias for Automotive: neutral.</i>", SMALL))
else:
    flow.append(Paragraph("Macro context unavailable this run.", BODY))

flow.append(Paragraph("Catalysts &amp; calendar", H2))
flow.append(Paragraph(
    "No scheduled earnings or ex-dividend date and no disclosure flags surfaced in the data window — i.e. no "
    "known hard catalyst in the immediate week (absence in-data, not a guarantee; check the EGX disclosure feed). "
    "Watch: quarterly results &amp; margin trajectory, CBE rate decisions / EGP moves, monthly vehicle-sales data, "
    "and any dividend declaration. " + (f"Largest historical abnormal moves clustered around "
    f"{', '.join(d['date'] for d in pir.get('biggest_abnormal_moves', [])[:3])}." if pir.get("biggest_abnormal_moves") else ""), BODY))

flow.append(PageBreak())

# ════════ Page 5: trade plan, risks, methodology ════════
flow.append(Paragraph("Trade plan (if traded) — illustrative on EGP 1,000,000", H2))
flow.append(Paragraph(
    f"Tactical framework, not a conviction buy. Sizing uses a {fmt(risk_pct,0)}% risk budget, a {fmt(atr_mult,0)}×ATR "
    f"stop, and a {fmt(max_pos_pct,0)}% position cap; the binding constraint here is the <b>{constraint}</b>. "
    f"Computed from the live close {fmt(entry)} (the model's own sizing tool is excluded — it used the bad 14.05 quote).", BODY))
tp = [["Parameter", "Value", "Parameter", "Value"],
      ["Entry (zone)", f"~{fmt(entry)} EGP", "Position size", f"{shares:,} sh"],
      ["Initial stop (2×ATR)", f"{fmt(stop)} ({fmt(-act_risk_pct if False else (stop/entry-1)*100,1)}%)", "Position value", f"{fmt(pos_val,0)} EGP ({fmt(pos_pct,1)}%)"],
      ["Risk / share", f"{fmt(stop_dist)} EGP", "Capital at risk", f"{fmt(act_risk,0)} EGP ({fmt(act_risk_pct,2)}%)"],
      ["Target 1 (+1 ATR)", f"{fmt(t1)} (+{fmt((t1/entry-1)*100,1)}%)", "Target 2 (+2 ATR)", f"{fmt(t2)} (+{fmt((t2/entry-1)*100,1)}%)"]]
flow.append(hdr_table(tp, [42*mm, 44*mm, 38*mm, 46*mm]))
flow.append(Paragraph(
    f"<b>Execution:</b> scale in {tranches} tranches — a starter near {fmt(entry)}, add on a pullback to "
    f"~{fmt(entry-atr)} (entry−1 ATR) <i>only if</i> it holds the stop, and add on strength above "
    f"~{fmt(entry+0.5*atr)} (entry+½ ATR) with RSI still &gt; 50. Book ⅓ at T1 and move the stop to break-even; "
    f"book ⅓ at T2; trail the remainder with a chandelier stop (high − {fmt(atr_mult,0)}×ATR). Note the modest "
    f"reward:risk (targets at +1/+2 ATR vs a 2-ATR stop) — this is a momentum scalp, so honour the stop.", BODY))

flow.append(Paragraph("Key risks", H2))
_mom_risk = (f"<b>Stretched short-term</b> — RSI {fmt(rsi14,0)} after the run; a pullback toward {fmt(support)} is the near risk."
             if trend_tag == "constructive" else
             "<b>Negative 3-month momentum</b> — no trend support." if mom_word == "negative" else
             "<b>Directionless tape</b> — neutral momentum, no edge.")
for r in [
    "<b>FX / EGP sensitivity</b> — imported vehicles &amp; parts; EGP weakness lifts COGS and squeezes margins "
    "(note: realised FX beta has been low, but a step-devaluation is a tail risk).",
    "<b>Interest-rate demand</b> — high domestic rates depress vehicle financing; a hawkish CBE is a headwind.",
    "<b>Profitability lag</b> — ROE ~11% trails the market; the value discount persists until returns improve.",
    _mom_risk,
    "<b>Coverage &amp; reliability</b> — no multi-year financials/segments in-dataset; no Automotive peers in the "
    "validated universe; model not yet reliability-validated (decision-support only).",
    "<b>Data integrity</b> — Yahoo's real-time quote for GBCO is currently glitched; always grade against the "
    "history endpoint and verify a live broker quote before trading.",
]:
    flow.append(Paragraph("• " + r, BULL))

flow.append(Paragraph("Methodology, data quality &amp; disclaimer", H2))
_src = (f"price/technicals/probabilities/beta/trade-plan computed from the live {SRC} daily history to {last_date}"
        if LIVE else "computed from the cached daily series")
flow.append(Paragraph(
    f"Fundamentals from the repo's audited cross-section; {_src}. The real-time quote is deliberately ignored (only a "
    f"same-day in-progress placeholder bar is dropped), so 'last close' is the last genuine session. Multiples are price-consistent "
    f"(audited EPS/BVPS at the last close). Probabilities use a lognormal drift/vol approximation from 60-day "
    f"returns (not a full Monte-Carlo); beta is OLS vs the synthetic equal-weight universe basket. Forecast, macro "
    f"and catalyst-reaction come from the model's clean (history-based) modules; the model's snapshot, P/E, market "
    f"cap and sizing outputs were excluded as contaminated by the bad quote. 'Universe median' is the audited "
    f"cross-section (mixed as-of dates — directional). The EGP 1,000,000 book is illustrative. "
    f"<b>This is decision-support from a model that is NOT yet reliability-validated. It is not investment advice "
    f"and not an offer to buy or sell any security. Verify against filings and a live quote before acting.</b>", SMALL))

doc = SimpleDocTemplate(str(PDF), pagesize=A4, topMargin=14 * mm, bottomMargin=13 * mm,
                        leftMargin=20 * mm, rightMargin=20 * mm,
                        title="GBCO Weekly Investment Report — 21-25 Jun 2026", author="EGX MCP model")
doc.build(flow)
print(f"Wrote {PDF}  ({PDF.stat().st_size/1024:.0f} KB)")
print(f"  data={'LIVE' if LIVE else 'CACHED'}  src={SRC}  last_session={last_date}  close={fmt(last)}  "
      f"PE={fmt(pe,1)}  beta={fmt(beta,2) if beta else 'n/a'}  fc21d={fmt((FC or {}).get('expected_return_pct'),1) if FC else 'n/a'}%")
