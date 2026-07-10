"""Tunable model parameters — the part the model is allowed to LEARN.

The decision layer reads its verdict thresholds from here instead of
hardcoding them, so a data-driven, human-approved update can change behavior
without a code edit. Defaults are the original hardcoded values; if a
repo-root model_params.json exists (written by `scripts/learn.py --apply`
after you approve a proposal), it overrides the defaults.

This is the SAFE form of "learning": parameters change only through an
explicit, OOS-validated, human-approved step — never silently at runtime.
Nothing here retrains on the fly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("egx-mcp.model_params")

_PARAMS_FILE = Path(__file__).parent.parent.parent / "model_params.json"

# Original hardcoded values — the baseline the model starts from.
DEFAULTS: dict[str, Any] = {
    "verdict_thresholds": {"BUY": 75, "ACCUMULATE": 65, "HOLD": 50, "REDUCE": 35},
    "score_weights": {"valuation": 0.30, "quality": 0.25, "momentum": 0.25, "risk": 0.20},
    "version": "default",
    "learned_at": None,
    "provenance": "hardcoded baseline",
}

_WEIGHT_KEYS = ("valuation", "quality", "momentum", "risk")


def _weights_ok(w: Any) -> bool:
    """Sane composite weights: all four factors, each positive, sum ≈ 1."""
    if not isinstance(w, dict):
        return False
    vals = [w.get(k) for k in _WEIGHT_KEYS]
    return (all(isinstance(v, (int, float)) and v > 0 for v in vals)
            and abs(sum(vals) - 1.0) <= 0.02)

_cache: dict[str, Any] | None = None


def load_params() -> dict[str, Any]:
    """Return active params: repo-root model_params.json if present, else DEFAULTS.

    Cached per process. Validates the loaded file has sane, monotone
    thresholds; falls back to DEFAULTS on anything malformed (never lets a
    bad learned file silently break verdicts)."""
    global _cache
    if _cache is not None:
        return _cache
    params = dict(DEFAULTS)
    if _PARAMS_FILE.exists():
        try:
            loaded = json.loads(_PARAMS_FILE.read_text(encoding="utf-8"))
            th = loaded.get("verdict_thresholds", {})
            keys = ("BUY", "ACCUMULATE", "HOLD", "REDUCE")
            vals = [th.get(k) for k in keys]
            th_ok = (all(isinstance(v, (int, float)) for v in vals)
                     and vals == sorted(vals, reverse=True)
                     and all(0 < v < 100 for v in vals))
            # score_weights is optional in a learned file; validate only if present.
            w_ok = ("score_weights" not in loaded) or _weights_ok(loaded["score_weights"])
            if th_ok and w_ok:
                params.update(loaded)
            elif not th_ok:
                log.warning("model_params.json has invalid thresholds %s — using defaults", th)
            else:
                log.warning("model_params.json has invalid score_weights %s — using defaults",
                            loaded.get("score_weights"))
        except Exception as e:  # noqa: BLE001
            log.warning("failed to read model_params.json (%s) — using defaults", e)
    _cache = params
    return params


def save_params(params: dict[str, Any]) -> None:
    """Write active params (called by the approved-apply path in learn.py)."""
    global _cache
    _PARAMS_FILE.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    _cache = None  # force reload


def thresholds() -> dict[str, float]:
    return load_params()["verdict_thresholds"]


def score_weights() -> dict[str, float]:
    """Composite sub-score weights — learnable, human-approved (see learn.py)."""
    return load_params().get("score_weights", DEFAULTS["score_weights"])
