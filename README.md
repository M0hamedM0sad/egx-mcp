# EGX MCP Server

A Model Context Protocol server for the Egyptian Exchange (EGX) that delivers **immediate, auditable trading decisions** — not just raw data. Built for FP&A research, equity analysis, and finance education in the MENA market.

## What it does

Sixteen tools, no API keys required. The headline tool is `decide(ticker)` — it returns a BUY/HOLD/SELL verdict with conviction, fair value, stop loss, target, and full rationale.

### Raw data (the foundation layer)

| Tool | Purpose |
|---|---|
| `get_quote(ticker)` | Live quote — price, change, volume, market cap, P/E |
| `get_history(ticker, period)` | OHLCV history with return / drawdown / volatility |
| `get_index(name)` | EGX30 / EGX70 / EGX100 with YTD return |
| `list_egx_stocks(sector?)` | Browse the curated EGX universe |
| `screen_stocks(...)` | Filter by P/E, market cap, volume, sector |
| `get_disclosures(ticker, days)` | Official disclosures from egx.com.eg |
| `get_news(ticker, lang)` | News from Yahoo Finance (EN) or Mubasher (AR) |
| `compute_technicals(ticker)` | RSI, MACD, MAs, Bollinger, ATR + signals |
| `portfolio_summary(csv_path?)` | Read CSV, compute live P&L per position |

### Decision layer (the new tools)

| Tool | Purpose |
|---|---|
| `get_fundamentals(ticker)` | **Sanitized** P/E, P/B, ROE, margin, leverage — auto-fixes Yahoo's bogus EGX P/E values |
| `get_macro_context()` | EGP/USD, Brent, gold, CBE rates, regime classification |
| `score_stock(ticker)` | Composite 0-100 score — valuation 30% / quality 25% / momentum 25% / risk 20% |
| `compare_peers(ticker)` | Sector ranking with relative-value verdict |
| `position_size(ticker, ...)` | ATR-based shares + stop + target for a given portfolio NAV |
| `get_catalyst_calendar(ticker)` | Forward earnings, ex-div, disclosure clusters (with blocking flags) |
| **`decide(ticker, ...)`** | **Headline tool** — synthesizes everything into BUY / ACCUMULATE / HOLD / REDUCE / AVOID with conviction, fair value, levels, and reasoning |

### How `decide()` works

1. Pulls **sanitized fundamentals** (corrects Yahoo's bad P/E by recomputing from price ÷ EPS).
2. Computes a **composite score** across valuation / quality / momentum / risk (each fully audited with notes).
3. Applies a **macro adjustment** of ±5 points based on sector fit with the current regime (e.g. tight CBE corridor favors banks, hurts real estate).
4. Ranks the name against **sector peers** to decide conviction.
5. Checks the **catalyst calendar** — earnings within 7 days is a *blocking* signal that downgrades BUY → HOLD.
6. Computes a **fair value estimate** = trailing EPS × sector median P/E (capped at ±50% from spot).
7. If portfolio NAV is provided and the verdict is buy-side, returns a **sized position** with explicit entry, stop, target, and shares.

The output includes `key_drivers`, `key_risks`, `blocking_catalysts`, and `data_quality_notes` so every verdict is auditable.

## Data sources

- **`yfinance`** — Yahoo Finance covers EGX with the `.CA` suffix (e.g. `CIRA.CA`, `^CASE30`). Free, no key, ~15-min delay.
- **`egx.com.eg`** — Official disclosures, scraped (subject to layout changes).
- **`mubasher.info`** — Arabic news, scraped.

There is **no public Thndr API** — see the project context if curious why this MCP doesn't talk to your brokerage account directly.

## Installation

```bash
# 1. Clone or copy the repo to your machine
cd /path/to/egx-mcp

# 2. Create a virtualenv (recommended — isolates yfinance from your other tools)
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. Install
pip install -e .
# or
pip install -r requirements.txt
```

## Verify it works (before touching Claude Desktop)

```bash
python -m tests.smoke
```

You should see live quotes for CIRA, COMI, SWDY and a portfolio P&L summary. If the smoke test passes, the MCP itself will work — any subsequent issues are config issues.

## Connect to Claude Desktop

Open the config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add the `egx` server. **Use full paths** — Claude Desktop launches with a minimal `PATH` and will silently fail on relative names like `python` or `python3`.

```json
{
  "mcpServers": {
    "egx": {
      "command": "/full/path/to/egx-mcp/.venv/bin/python",
      "args": ["-m", "egx_mcp.server"],
      "cwd": "/full/path/to/egx-mcp",
      "env": {
        "EGX_PORTFOLIO_CSV": "/full/path/to/your/portfolio.csv",
        "PYTHONPATH": "/full/path/to/egx-mcp"
      }
    }
  }
}
```

On Windows, the `command` would be e.g. `C:\\Users\\Mohamed\\egx-mcp\\.venv\\Scripts\\python.exe` (note the double backslashes).

Then **fully quit and restart Claude Desktop** (right-click the dock icon → Quit, not just close the window).

## Portfolio CSV format

Required columns (case-insensitive): `ticker`, `shares`, `cost_basis` (in EGP per share).
Optional: `purchase_date`, `account`, `notes`.

```csv
ticker,shares,cost_basis,purchase_date,notes
CIRA,500,12.50,2024-09-15,Education sector
COMI,100,72.30,2024-11-02,Long-term hold
SWDY,200,4.85,2025-01-20,
```

A working sample is included as `sample_portfolio.csv`.

## Example prompts to try in Claude

**Decision-grade prompts (use the new tools):**
```
Should I buy COMI? My portfolio is 500,000 EGP and I risk 1% per trade.

Decide on CIRA, TMGH, and SWDY — give me a ranked verdict with conviction.

Show me macro context for EGX today. Which sectors are tailwinds vs headwinds?

Compare CIRA against its sector peers — is it the cheapest or the best?

What's the fair value of HRHO based on the sector median P/E?

Score every bank in my universe and give me the top 3 buys.
```

**Raw-data prompts (the foundation tools):**
```
What's COMI trading at and how does the chart look on a 6-month view?

Run a screen for EGX banks with P/E below 10 and market cap above 10 billion EGP.

Show me my portfolio with live P&L. Which positions are dragging?

What disclosures came out this week for CIRA?

قارن أداء الشركات العقارية في البورصة المصرية على مدى الـ 6 شهور الماضية.
```

## Architecture

```
egx_mcp/
├── server.py              # FastMCP entry point — tool definitions only
└── data/
    ├── universe.py        # Curated EGX ticker map + resolver
    ├── market.py          # yfinance wrapper with TTL cache
    ├── disclosures.py     # egx.com.eg scraper
    ├── news.py            # Yahoo + Mubasher
    ├── technicals.py      # Pure-pandas indicators
    ├── portfolio.py       # CSV reader + live P&L
    │
    ├── fundamentals.py    # Sanitized P/E, P/B, ROE + sector medians
    ├── macro.py           # EGP/USD, Brent, CBE rates, sector macro bias
    ├── scoring.py         # Composite 0-100 with audit trail
    ├── peers.py           # Sector relative ranking
    ├── sizing.py          # ATR-based position sizing
    ├── calendar.py        # Earnings + ex-div + disclosure clustering
    └── decision.py        # Synthesizer — BUY/HOLD/SELL with rationale
```

Each adapter is independent. To swap a data source (say, replace yfinance with EGXPY or ICE), edit only that one module — `server.py`, the tool contracts, and the decision layer are untouched.

## Caveats (read these — they're what actually breaks in practice)

- **Yahoo's EGX feed is ~15-min delayed.** Fine for research, not for execution.
- **Yahoo has very limited history for EGX indices.** `^CASE30`, `EGX100.CA`, etc. typically return only the latest bar via `yfinance.history()`, even though the live quote works fine. So `get_index()` gives you the current value but `ytd_return_pct` will often be null. For historical index analysis, use the EGX 30 ETF (`EGS69491M015.CA`) as a proxy or compute a custom basket from the constituents.
- **Disclosures scraper is fragile.** EGX changes their HTML occasionally; if `get_disclosures` returns errors, check `_DISCLOSURES_URL` and the parsing logic in `disclosures.py`.
- **Curated universe.** Only ~30 of EGX's ~240 listings are mapped by nickname. Add yours to `universe.py` → `EGX_UNIVERSE` as needed. ISIN-style codes like `EGS65541C012` work without registration.
- **Screener is slow.** ~30s for the full universe because each row hits yfinance. For the AI for Finance course, consider pre-warming the cache or caching to disk.
- **`pe_ratio` from Yahoo can be wrong for EGX names.** The smoke test caught CIRA returning a P/E of 0.12 — that's a Yahoo data quality issue, not an MCP bug. Cross-check P/E from EGX official disclosures or Mubasher before using in commentary.

## Extending

Common additions:

- **Pre-warm cache job** — run `screen_stocks()` nightly via cron and pickle results.
- **Sector benchmarks** — add a `compare_to_sector(ticker)` tool.
- **Earnings calendar** — scrape from Mubasher's earnings page.
- **CBE rates / FX** — add a `get_egp_rates()` tool fed from CBE's daily fixing.
- **Power BI export** — add a `to_powerbi(format='parquet')` tool that writes to your Fabric lakehouse.

## License

Personal use. Not investment advice. Always verify quotes against the EGX official tape before acting.
