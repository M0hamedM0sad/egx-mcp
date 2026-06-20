# EGX MCP — Model Reliability Roadmap

How we go from "the model produces verdicts" to "we can rely on the model's
judgment." The guiding principle is **evidence before weighting**: prove a
signal has edge before you let it move a verdict. A more confident black box
is not a more trustworthy one.

The work splits into two tracks. **Track A earns trust** (measurement and
validation — mostly our own data). **Track B upgrades the signals** (where
Hugging Face models help). Do A first: B improves the inputs, but A is what
tells you whether those improvements are real.

> **Why this order matters.** The full live model (fundamentals + macro +
> catalysts + sentiment) is *not* historically replayable — those inputs are
> "current", so a backtest of them carries lookahead bias. The only honest
> way to measure the full model is to grade its verdicts *forward*. That's
> A1, and it only becomes statistically meaningful as briefings accumulate
> (aim for **30+ graded calls per bucket**).

---

## Track A — Earn trust

### A1 · Grade verdicts forward → the evidence base
`python -m tests.grade_briefings`

Reads `briefings/*.json`, pulls realized prices forward, grades every verdict
(v8b + weekly picks) against EGX 30 over 5- and 21-day horizons. Writes
`logs/graded_verdicts.{jsonl,csv}`. Un-elapsed calls are stored as `pending`
and graded automatically on the next run.

- **Re-run after every briefing.** The dataset compounds; this is the asset.
- **Correct = beat the index** for buy-side, lag it for sell-side. (EGX is in
  a strong bull — absolute "went up" flatters everything, so we grade excess.)
- **Version it on Hugging Face** once it's meaningful:
  `ds.push_to_hub('m0hamedm0sad/egx-verdict-outcomes')` — reproducible, and it
  becomes training data for B4.

### A2 · Calibration — does conviction mean anything?
`python -m tests.calibration_report`

Consumes A1's output. Checks whether "high conviction" actually hits more than
"low". A model that's 60% accurate *and knows when it's unsure* beats a 70%
one that's confidently wrong. If conviction isn't monotone with accuracy, the
conviction logic needs recalibration before we trust it.

### A3 · Out-of-sample & regime robustness (price core)
`python -m tests.backtest_accuracy --holdout-months 10`

Runnable today on real history (the price-momentum core is point-in-time
clean). Reports directional accuracy per verdict band, **in-sample vs
out-of-sample**, and **bull vs correction**. OOS holding near IS is the trust
signal; a bull-only edge is the #1 way EGX models blow up.

### A4 · Statistical validation — real edge or luck?
`python -m tests.validate_edge`

Consumes A1's graded verdicts and renders one honest verdict
(EDGE_CONFIRMED / INCONCLUSIVE / NEGATIVE_EDGE / INSUFFICIENT_SAMPLE). Fixes
two traps the flat scorecard misses: (1) the **sign-convention trap** — it
grades the *direction-aware signed edge*, not raw excess (a wrong sell whose
stock soars no longer flatters the headline); (2) **false sample size** — rows
are clustered by briefing date (the same call is graded at 5d *and* 21d, and
calls share a market day), so the CI is a **cluster bootstrap** over distinct
dates and the binomial test runs on the real n. Exit code is 0 only when an
edge is actually confirmed, so it can gate automation.

### A0 · Transport reliability — does the MCP itself behave?
`python -m tests.validate_mcp_tools`

The prerequisite gate: enumerates every registered MCP tool and checks the
three contracts a client depends on — **never raises**, **JSON-serializable**,
**returns a dict** — classifying each as OK / DEGRADED (graceful failure) /
SLOW (heavy batch tool, valid output but >timeout) / CRASH / NONSERIAL /
TIMEOUT (genuine hang). Only CRASH/NONSERIAL/NONDICT/TIMEOUT fail the run; SLOW
and DEGRADED are notes. A verdict you can't reliably deliver is worthless
regardless of its edge, so run this before trusting any of A1–A4.

---

## Track B — Upgrade the signals (Hugging Face)

Every Track-B model is an **optional dependency with a graceful fallback** —
the core MCP stays zero-dependency. Activate per area as you validate it.

### B1 · Sentiment: lexicon → transformer
`pip install 'egx-mcp[sentiment]'` → set `EGX_SENTIMENT_BACKEND=transformer`

- **EN:** [ProsusAI/finbert](https://hf.co/ProsusAI/finbert)
- **AR:** [CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment](https://hf.co/CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment) (dialectal — fits Mubasher register)

**Prove it before flipping the default:**
1. `python -m tests.make_labeled_set COMI CIRA SWDY --balance 15` → labeling stub
2. Label the `label` column, then `python -m tests.eval_sentiment <file.csv>`
3. Flip the default only if the transformer wins by a meaningful margin.

### B2 · Disclosure event extraction
`pip install 'egx-mcp[sentiment]'` · tool: `tag_disclosure_events`

Zero-shot multilingual NLI ([joeddav/xlm-roberta-large-xnli](https://hf.co/joeddav/xlm-roberta-large-xnli))
tags each disclosure into a fixed event taxonomy (dividend, capital increase,
M&A, profit warning, earnings, board change, trading suspension) with a
`material` flag the catalyst/decision layer can gate a BUY on. Falls back to an
AR+EN keyword classifier when the model isn't installed.

### B3 · Probabilistic price forecast
`pip install 'egx-mcp[forecast]'` · tool: `forecast_price`

Chronos / [TimesFM](https://hf.co/google/timesfm-2.5-200m-transformers) return a
forecast *distribution* → `expected_return_pct` + `uncertainty_pct` (q90–q10).
**Use as one input, not an oracle** — EGX is thin and FX-driven; backtest the
signal in A3 before weighting it. A wide uncertainty band argues for sizing
down, which is often more robust than trading the point forecast.

### B4 · Domain fine-tune (only after you have labels)
`pip install 'egx-mcp[train]'` → `python scripts/finetune_classifier.py ...`

Once A1 has produced enough labeled examples (or a merged headline set, **≥500
rows balanced across classes**), fine-tune a small efficient model on EGX
register — [ModernBERT](https://hf.co/answerdotai/ModernBERT-base) base, the
pattern behind [tsphua/modernbert-fingpt](https://hf.co/tsphua/modernbert-fingpt).
**Do not do this first** — fine-tuning before you have EGX-specific labels just
overfits to generic US-market sentiment. A no-code alternative: upload the same
dataset to Hugging Face AutoTrain.

---

## Datasets (for B1 tuning / B4)

| Dataset | Use |
|---|---|
| [takala/financial_phrasebank](https://hf.co/datasets/takala/financial_phrasebank) | EN finance sentiment (the FinBERT training set) |
| [zeroshot/twitter-financial-news-sentiment](https://hf.co/datasets/zeroshot/twitter-financial-news-sentiment) | EN retail-tone (closer to Mubasher register) |
| [drelhaj/Arabic-news-and-management-corpus](https://hf.co/datasets/drelhaj/Arabic-news-and-management-corpus) | AR economics/financial news — closest to EGX press |

The highest-value dataset is the one **we** build: `logs/graded_verdicts.jsonl`
(A1) and our own labeled EGX headlines (make_labeled_set + review).

---

## What "relying on the model" requires

A reliable model = **measured edge that survives out-of-sample (A3),
calibrated to conviction (A2), holds across regimes (A3), and is gated by a
human.** Hugging Face gives us better *inputs* (B1–B4). Track A gives us the
*proof*. We need both — but A is the gate, and HF can't do A for us; it can
only host the evidence.

## Recommended sequence

1. **A1** every briefing — start accumulating evidence now.
2. **A3** — validate the price core OOS/regime (runnable today).
3. **B1** — install, label, eval; flip sentiment default if it wins.
4. **B2** — wire material events into the catalyst gate.
5. **A2** — once ~30+ graded calls exist, check calibration.
6. **B3** — add forecast as a feature, backtest it before weighting.
7. **B4** — fine-tune once labels are plentiful.

---
*Not investment advice. Every signal is auditable by design; verify material
calls against the official EGX tape before acting.*
