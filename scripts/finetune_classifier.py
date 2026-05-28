"""B1 / B4 — Fine-tune a text classifier on your labeled EGX data.

One script, two uses:

  B1  Adapt sentiment to EGX register. Start from a finance/Arabic base and
      fine-tune on headlines you labeled with tests/make_labeled_set.py +
      manual review:
        python scripts/finetune_classifier.py \
            --base ProsusAI/finbert \
            --data tests/labeled_set_stub.csv \
            --out models/egx-finbert
      (For Arabic, --base CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment.)

  B4  Domain fine-tune a small, efficient model on a larger labeled corpus
      (e.g. once tests/grade_briefings.py has produced enough examples, or a
      merged headline set). ModernBERT is a strong, fast base — the same one
      behind tsphua/modernbert-fingpt:
        python scripts/finetune_classifier.py \
            --base answerdotai/ModernBERT-base \
            --data data/egx_labeled.csv \
            --out models/egx-modernbert --epochs 4

DO THIS ONLY AFTER you have labeled data. Fine-tuning before you have
EGX-specific labels just overfits to generic US-market sentiment — the whole
point is to teach the model Mubasher/EGX register. Aim for >=500 labeled rows,
balanced across classes (see make_labeled_set.py --balance).

CSV columns: text, label (negative|neutral|positive). `lang` optional/ignored.

Needs the training extras (not part of the core MCP):
    pip install 'egx-mcp[train]'      # transformers, torch, datasets, scikit-learn

This script is intentionally a thin, auditable Trainer loop. For a no-code
alternative, the same dataset uploads straight to Hugging Face AutoTrain.
"""
from __future__ import annotations

import argparse
import sys

_LABELS = ["negative", "neutral", "positive"]
_LABEL2ID = {l: i for i, l in enumerate(_LABELS)}


def _require_deps():
    try:
        import numpy  # noqa: F401
        import torch  # noqa: F401
        from datasets import Dataset  # noqa: F401
        from sklearn.metrics import accuracy_score, f1_score  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSequenceClassification, AutoTokenizer,
            Trainer, TrainingArguments,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Training deps missing ({e}).\nInstall with: pip install 'egx-mcp[train]'")
        sys.exit(1)


def _load_rows(path: str):
    import csv
    rows, skipped = [], 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            text = (r.get("text") or "").strip()
            label = (r.get("label") or "").strip().lower()
            if text and label in _LABEL2ID:
                rows.append({"text": text, "label": _LABEL2ID[label]})
            else:
                skipped += 1
    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune a text classifier (B1/B4).")
    ap.add_argument("--base", required=True, help="HF base model id.")
    ap.add_argument("--data", required=True, help="Labeled CSV (text,label).")
    ap.add_argument("--out", required=True, help="Output directory for the fine-tuned model.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _require_deps()
    import numpy as np
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        Trainer, TrainingArguments,
    )

    rows, skipped = _load_rows(args.data)
    if len(rows) < 50:
        print(f"Only {len(rows)} usable rows (skipped {skipped}). Need more labeled data "
              "to fine-tune meaningfully — aim for >=500. Aborting.")
        return 1
    print(f"Loaded {len(rows)} labeled rows (skipped {skipped}). Base: {args.base}")

    # Stratified-ish split via shuffle (datasets handles class spread well enough
    # at this size; for tiny sets the val metric is noisy regardless).
    ds = Dataset.from_list(rows).shuffle(seed=args.seed)
    split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)

    tok = AutoTokenizer.from_pretrained(args.base)

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_length)

    split = split.map(_tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(_LABELS),
        id2label={i: l for l, i in _LABEL2ID.items()}, label2id=_LABEL2ID,
        ignore_mismatched_sizes=True,  # base may have a different head size
    )

    def _metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=split["train"], eval_dataset=split["test"],
        tokenizer=tok, compute_metrics=_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(f"\nValidation: accuracy={metrics.get('eval_accuracy'):.3f}  "
          f"f1_macro={metrics.get('eval_f1_macro'):.3f}")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved fine-tuned model -> {args.out}")
    print("Use it by setting the env var, then run the sentiment eval to confirm it wins:")
    print(f"    EGX_FINBERT_MODEL={args.out}   (or EGX_ARABIC_SENTIMENT_MODEL)")
    print("    python -m tests.eval_sentiment <your_labeled.csv>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
