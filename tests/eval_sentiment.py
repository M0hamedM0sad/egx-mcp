"""Compare the lexicon vs. transformer sentiment backends on labeled headlines.

Scores each headline OFFLINE with both backends (no network / yfinance), maps
the signed score to a 3-class label, and reports per-backend accuracy so you
can decide whether the transformer upgrade is worth the load + inference cost
for EGX flow before wiring it into decide().

    python -m tests.eval_sentiment                       # bundled EGX sample
    python -m tests.eval_sentiment path/to/labeled.csv   # your own headlines

CSV columns: lang (en|ar), label (positive|negative|neutral), text.

The transformer backend needs the optional deps:
    pip install 'egx-mcp[sentiment]'
If they're missing it's reported as "unavailable" and only the lexicon runs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Arabic headlines need a UTF-8 console (Windows defaults to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from egx_mcp.data import sentiment, transformer_sentiment

# Map a signed score in [-1, +1] to a 3-class label. The 0.1 deadzone mirrors
# the mildly_bullish / mildly_bearish thresholds in sentiment._label.
_DEADZONE = 0.1


def _to_label(score: float) -> str:
    if score >= _DEADZONE:
        return "positive"
    if score <= -_DEADZONE:
        return "negative"
    return "neutral"


def _load(path: Path) -> list[dict[str, str]]:
    # utf-8-sig strips a BOM if present — the stub generator writes one for
    # Excel, and Excel re-adds one when you save. Without this the first
    # header becomes "﻿lang" and row["lang"] KeyErrors.
    with path.open(encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if (row.get("text") or "").strip()]


def _eval_backend(rows: list[dict[str, str]], backend: str) -> dict:
    """Score every row with one backend. Returns metrics + per-row predictions."""
    correct = 0
    confusion: dict[tuple[str, str], int] = {}
    preds: list[dict] = []

    for row in rows:
        lang = (row["lang"] or "").strip().lower()
        gold = (row["label"] or "").strip().lower()
        text = row["text"].strip()

        if backend == "transformer":
            score, _ = transformer_sentiment.score_text(text, lang)
        else:
            score, _ = sentiment._score_text(text, lang)

        pred = _to_label(score)
        if pred == gold:
            correct += 1
        confusion[(gold, pred)] = confusion.get((gold, pred), 0) + 1
        preds.append({"lang": lang, "gold": gold, "pred": pred,
                      "score": round(score, 3), "text": text})

    n = len(rows)
    return {
        "backend": backend,
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "correct": correct,
        "confusion": confusion,
        "preds": preds,
    }


def _print_report(res: dict, show_misses: bool = True) -> None:
    print(f"\n{'=' * 64}")
    print(f"Backend: {res['backend']}   "
          f"accuracy = {res['accuracy']:.1%}  ({res['correct']}/{res['n']})")
    print('=' * 64)

    labels = ["positive", "negative", "neutral"]
    print(f"{'gold \\ pred':>14} | " + " ".join(f"{l[:4]:>5}" for l in labels))
    print("-" * 44)
    for g in labels:
        row = " ".join(f"{res['confusion'].get((g, p), 0):>5}" for p in labels)
        print(f"{g:>14} | {row}")

    if show_misses:
        misses = [p for p in res["preds"] if p["pred"] != p["gold"]]
        if misses:
            print(f"\n  Misses ({len(misses)}):")
            for m in misses:
                print(f"   [{m['lang']}] gold={m['gold']:>8} pred={m['pred']:>8} "
                      f"score={m['score']:+.2f}  {m['text'][:70]}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).parent / "sample_sentiment_labeled.csv")
    if not path.exists():
        print(f"Labeled CSV not found: {path}")
        return 1

    rows = _load(path)
    print(f"Loaded {len(rows)} labeled headlines from {path.name}")
    by_lang: dict[str, int] = {}
    for r in rows:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
    print(f"By language: {by_lang}")

    lex = _eval_backend(rows, "lexicon")
    _print_report(lex)

    en_ok = transformer_sentiment.available("en")
    ar_ok = transformer_sentiment.available("ar")
    print(f"\nTransformer availability — EN: {en_ok}, AR: {ar_ok}")
    if en_ok or ar_ok:
        tf = _eval_backend(rows, "transformer")
        _print_report(tf)
        delta = tf["accuracy"] - lex["accuracy"]
        print(f"\n{'#' * 64}")
        print(f"# Transformer vs lexicon: {delta:+.1%} accuracy "
              f"({tf['accuracy']:.1%} vs {lex['accuracy']:.1%})")
        print('#' * 64)
    else:
        print("Transformer backend unavailable — install with: "
              "pip install 'egx-mcp[sentiment]'")
        print("Lexicon-only result reported above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
