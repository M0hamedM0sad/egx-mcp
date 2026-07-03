"""Daily EGX levels email — the >60% up-probability trade-levels table.

Standalone and lean: uses ONLY the glitch-guarded investing.com pipeline plus
TradingView for liquidity, so it doesn't depend on the Yahoo-backed parts of
daily_briefing.py. Designed to run pre-market (08:30 Cairo) on EGX trading
days (Sun-Thu). Self-skips Fri/Sat.

For each liquid name (turnover >= EGX_MIN_TURNOVER_M, default 5M EGP) whose
model up-probability exceeds EGX_MIN_PROB_UP (default 0.60), the table gives:
entry (last close), pivot support/resistance, R2, a tight 1-day stop, and the
2xATR swing stop.

Always writes briefings/levels_YYYY-MM-DD.html. Emails it if SMTP creds exist
(accepts either SMTP_* or GMAIL_* env conventions).

    python -m scripts.levels_email           # run (skips Fri/Sat)
    python -m scripts.levels_email --force    # run regardless of weekday
"""
from __future__ import annotations

import io
import os
import smtplib
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

from egx_mcp.data._certs import ensure_ca_bundle

ensure_ca_bundle()
from curl_cffi import requests as cr

from egx_mcp.data import simulation, technicals, investing

MIN_PROB_UP = float(os.environ.get("EGX_MIN_PROB_UP", "0.60"))
MIN_TURNOVER_M = float(os.environ.get("EGX_MIN_TURNOVER_M", "5"))
EGX_TRADING_DAYS = {6, 0, 1, 2, 3}  # Sun=6, Mon..Thu=0..3 (Python weekday)
OUTPUT_DIR = Path(__file__).parent.parent / "briefings"


def _tv_volume() -> dict[str, float]:
    for _ in range(4):
        try:
            d = cr.post("https://scanner.tradingview.com/egypt/scan",
                        json={"columns": ["close", "volume", "type"], "range": [0, 600]},
                        impersonate="chrome", timeout=40).json()
            return {x["s"].split(":")[1]: (x["d"][1] or 0)
                    for x in d.get("data", []) if x["d"][2] == "stock"}
        except Exception:
            time.sleep(2)
    return {}


def build_rows() -> tuple[list[dict], str]:
    vol = _tv_volume()
    out = simulation.scan_universe(horizon_days=1, n_paths=4000, lookback_days=60,
                                   min_prob_up_2pct=0.0, min_expected_return_pct=-100.0,
                                   seed=42, full_market=True)
    rows, asof = [], None
    for r in out["ranked"]:
        if r.get("prob_up", 0) <= MIN_PROB_UP:
            continue
        turnover_m = r["current_price"] * vol.get(r["ticker"], 0) / 1e6
        if turnover_m < MIN_TURNOVER_M:
            continue
        df = investing.daily_history(r["ticker"], lookback_days=40)
        if df.empty:
            continue
        last = df.iloc[-1]
        H, L, C = float(last["High"]), float(last["Low"]), float(last["Close"])
        P = (H + L + C) / 3
        atr = (technicals.compute(r["ticker"], period="6mo").get("indicators") or {}).get("atr_14") or 0
        asof = r.get("baseline_date")
        rows.append({
            "ticker": r["ticker"],
            "prob_up": r["prob_up"] * 100,
            "entry": round(C, 2),
            "support": round(2 * P - H, 2),
            "resistance": round(2 * P - L, 2),
            "next_r": round(P + (H - L), 2),
            "stop_1d": round(2 * P - H - 0.01, 2),
            "stop_atr": round(C - 2 * atr, 2) if atr else None,
            "high_20d": round(float(df["High"].tail(20).max()), 2),
            "turnover_m": round(turnover_m, 1),
        })
    rows.sort(key=lambda x: x["prob_up"], reverse=True)
    return rows, asof or "n/a"


def render(rows: list[dict], asof: str) -> tuple[str, str]:
    sub = f"P(up) > {MIN_PROB_UP:.0%} · turnover ≥ {MIN_TURNOVER_M:.0f}M EGP · forecast for next session (anchored {asof})"
    if not rows:
        text = f"EGX levels — {asof}\n{sub}\n\nNo names cleared the filter today."
        return f"<html><body><h2>EGX levels</h2><p>{sub}</p><p>No names cleared the filter today.</p></body></html>", text

    cols = ["Stock", "P(up)", "Entry", "Support", "Resist", "Next R", "Stop 1d", "Stop ATR", "20d High"]
    th = "".join(f"<th style='padding:6px 10px;background:#f4f6f9;color:#0d3e66;text-align:left'>{c}</th>" for c in cols)
    trs = []
    for p in rows:
        cells = [p["ticker"], f"{p['prob_up']:.0f}%", f"{p['entry']:.2f}", f"{p['support']:.2f}",
                 f"{p['resistance']:.2f}", f"{p['next_r']:.2f}", f"{p['stop_1d']:.2f}",
                 f"{p['stop_atr']:.2f}" if p["stop_atr"] is not None else "—", f"{p['high_20d']:.2f}"]
        trs.append("<tr>" + "".join(f"<td style='padding:6px 10px;border-bottom:1px solid #eee'>{c}</td>" for c in cells) + "</tr>")
    html = (f"<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#1a1a1a'>"
            f"<h2 style='color:#0d3e66'>EGX trade-levels — {asof}</h2><p style='color:#555'>{sub}</p>"
            f"<table style='border-collapse:collapse'><tr>{th}</tr>{''.join(trs)}</table>"
            f"<p style='color:#888;font-size:12px;margin-top:14px'>Probability tilt, not advice. "
            f"Direction edge is modest; the stop is what protects the trade. Stop 1d = just below pivot "
            f"support; Stop ATR = entry − 2×ATR (swing).</p></body></html>")
    w = [f"EGX trade-levels — {asof}", sub, ""]
    w.append(f"{'Stock':<7}{'P(up)':>6}{'Entry':>9}{'Supp':>9}{'Resist':>9}{'NextR':>9}{'Stop1d':>9}{'StopATR':>9}{'20dHi':>9}")
    for p in rows:
        sa = f"{p['stop_atr']:.2f}" if p["stop_atr"] is not None else "-"
        w.append(f"{p['ticker']:<7}{p['prob_up']:>5.0f}%{p['entry']:>9.2f}{p['support']:>9.2f}"
                 f"{p['resistance']:>9.2f}{p['next_r']:>9.2f}{p['stop_1d']:>9.2f}{sa:>9}{p['high_20d']:>9.2f}")
    w += ["", "Probability tilt, not advice. Honor the stop."]
    return html, "\n".join(w)


def send_email(html: str, text: str, subject: str) -> bool:
    """Send via SMTP. Accepts SMTP_* or GMAIL_* env conventions."""
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = os.environ.get("SMTP_PORT", "587")
    user = os.environ.get("SMTP_USER") or os.environ.get("GMAIL_USER")
    password = os.environ.get("SMTP_PASS") or os.environ.get("GMAIL_APP_PASSWORD")
    to_addr = os.environ.get("BRIEFING_EMAIL_TO") or os.environ.get("RECIPIENT") or user
    if not all([user, password, to_addr]):
        print("SMTP creds not set — skipping email (saved file only).")
        print("  Set SMTP_USER/SMTP_PASS (or GMAIL_USER/GMAIL_APP_PASSWORD) and a recipient.")
        return False
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, user, to_addr
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


def main() -> int:
    force = "--force" in sys.argv
    if datetime.now().weekday() not in EGX_TRADING_DAYS and not force:
        print(f"{datetime.now():%A} is not an EGX trading day — skipping.")
        return 0
    rows, asof = build_rows()
    html, text = render(rows, asof)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fpath = OUTPUT_DIR / f"levels_{datetime.now():%Y-%m-%d}.html"
    fpath.write_text(html, encoding="utf-8")
    print(f"Saved {fpath} ({len(rows)} names)")
    send_email(html, text, f"EGX levels — {asof} ({len(rows)} names, P(up)>{MIN_PROB_UP:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
