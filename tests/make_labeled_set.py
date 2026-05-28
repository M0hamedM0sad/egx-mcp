"""Pull live EGX headlines into a CSV stub you can hand-label, then feed to
tests/eval_sentiment.py to measure lexicon vs. transformer on REAL flow.

    python -m tests.make_labeled_set                      # market-wide, AR+EN
    python -m tests.make_labeled_set COMI CIRA SWDY       # per-ticker
    python -m tests.make_labeled_set --lang ar --limit 25 --out mubasher.csv

The output has a blank `label` column for you to fill (positive | negative |
neutral) and a `lexicon_guess` column so you can review fast rather than start
from scratch — but label what YOU think, not what the lexicon guessed, or the
eval just measures the lexicon against itself.

Once labeled:
    python -m tests.eval_sentiment <out.csv>

Writes UTF-8-BOM so Excel opens Arabic correctly. Refuses to clobber an
existing file unless --force (don't lose in-progress labels).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from egx_mcp.data import news, sentiment

_FIELDS = ["lang", "label", "lexicon_guess", "ticker", "date", "source", "text", "url"]


def _guess(text: str, lang: str) -> str:
    """Lexicon's 3-class call — a starting point for review, not ground truth."""
    score, _ = sentiment._score_text(text, lang)
    if score >= 0.1:
        return "positive"
    if score <= -0.1:
        return "negative"
    return "neutral"


def _balance(rows: list[dict], cap: int) -> list[dict]:
    """Cap rows per `lexicon_guess` class so the set isn't all-neutral.

    Headline flow skews heavily neutral; an unbalanced set makes accuracy
    look high for free (predict neutral, win). Balancing on the lexicon's
    guess is a proxy — it doesn't peek at true labels (you haven't set them
    yet) — but it pulls in enough pos/neg candidates to label meaningfully.
    Insertion order is preserved within each class.
    """
    kept_per_class: dict[str, int] = {}
    out: list[dict] = []
    for r in rows:
        cls = r["lexicon_guess"]
        if kept_per_class.get(cls, 0) >= cap:
            continue
        kept_per_class[cls] = kept_per_class.get(cls, 0) + 1
        out.append(r)
    return out


def _dist(rows: list[dict]) -> dict[str, int]:
    d: dict[str, int] = {}
    for r in rows:
        d[r["lexicon_guess"]] = d.get(r["lexicon_guess"], 0) + 1
    return d


def _collect(tickers: list[str], langs: list[str], limit: int) -> list[dict]:
    """Fetch headlines for each (ticker, lang); dedupe on normalized title."""
    targets = tickers or [None]  # None = market-wide
    rows: list[dict] = []
    seen: set[str] = set()

    for tk in targets:
        for lang in langs:
            label = tk or "MARKET"
            try:
                payload = news.fetch(ticker=tk, lang=lang, limit=limit)
            except Exception as e:  # noqa: BLE001 — one source failing shouldn't abort
                print(f"  ! {label}/{lang}: fetch failed ({e})")
                continue

            arts = payload.get("articles", []) or []
            added = 0
            for art in arts:
                title = (art.get("title") or "").strip()
                if not title:
                    continue
                key = " ".join(title.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "lang": lang,
                    "label": "",  # for the human
                    "lexicon_guess": _guess(title, lang),
                    "ticker": tk or "",
                    "date": art.get("date") or "",
                    "source": art.get("source") or "",
                    "text": title,
                    "url": art.get("url") or "",
                })
                added += 1
            print(f"  {label}/{lang}: {added} new ({len(arts)} fetched)")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a labeling stub from live EGX headlines.")
    ap.add_argument("tickers", nargs="*", help="EGX codes/nicknames. Omit for market-wide.")
    ap.add_argument("--lang", choices=["en", "ar", "both"], default="both")
    ap.add_argument("--limit", type=int, default=20, help="Headlines per ticker per language.")
    ap.add_argument("--out", default=str(Path(__file__).parent / "labeled_set_stub.csv"))
    ap.add_argument("--balance", type=int, metavar="N",
                    help="Cap headlines per lexicon-guessed class (pos/neg/neutral) "
                         "so the set isn't dominated by neutral flow.")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing --out file.")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"Refusing to overwrite {out} (use --force). "
              "Protecting any labels already in it.")
        return 1

    langs = ["en", "ar"] if args.lang == "both" else [args.lang]
    print(f"Fetching headlines — tickers={args.tickers or ['MARKET']} "
          f"langs={langs} limit={args.limit}/each")

    rows = _collect(args.tickers, langs, args.limit)
    if not rows:
        print("No headlines collected — nothing written. "
              "(Network down, or sources returned empty for these names.)")
        return 1

    if args.balance:
        before = _dist(rows)
        rows = _balance(rows, args.balance)
        print(f"\nBalanced to <= {args.balance}/class (by lexicon guess): "
              f"{before} -> {_dist(rows)}")

    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_lang: dict[str, int] = {}
    for r in rows:
        by_lang[r["lang"]] = by_lang.get(r["lang"], 0) + 1
    print(f"\nWrote {len(rows)} unlabeled headlines to {out}")
    print(f"By language: {by_lang}")
    print("\nNext: fill the `label` column (positive|negative|neutral), then run:")
    print(f"    python -m tests.eval_sentiment {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
