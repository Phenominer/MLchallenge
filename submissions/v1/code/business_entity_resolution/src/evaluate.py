"""Validation split and the official macro F0.5 scorer.

Scorer: F0.5 per Source 1 entity, averaged over ALL entities in the evaluation set.
Singleton (no true matches): 1.0 if prediction is empty, else 0.0.
Non-singleton with empty prediction: precision undefined -> 0.0.
"""
import json

import numpy as np

from io_utils import CACHE, load_truth

SEED = 42


def f05(pred: set, true: set) -> float:
    """F0.5 for one Source 1 entity, including the singleton rule."""
    if not true:
        return float(not pred)
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def f05_macro(pred: dict, truth: dict) -> float:
    """Macro F0.5 over every entity in truth (missing predictions count as empty)."""
    return float(np.mean([f05(pred.get(k, set()), v) for k, v in truth.items()]))


def split_ids(dev_size: int = 50_000) -> dict:
    """Return {'train','val','dev'} lists of S1 ids: 80/20 split by entity, dev = sample of val."""
    path = CACHE / "split.json"
    if path.exists():
        return json.load(open(path))
    ids = np.array(sorted(load_truth()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(ids)
    cut = int(len(ids) * 0.8)
    val = ids[cut:]
    out = {"train": ids[:cut].tolist(), "val": val.tolist(), "dev": val[:dev_size].tolist()}
    json.dump(out, open(path, "w"))
    return out


if __name__ == "__main__":
    # README example: pred 3, truth 2, overlap 2 -> 0.714
    assert round(f05({"a", "b", "c"}, {"a", "c"}), 3) == 0.714
    assert f05(set(), set()) == 1.0 and f05({"a"}, set()) == 0.0 and f05(set(), {"a"}) == 0.0
    assert f05_macro({"x": {"a"}}, {"x": {"a"}, "y": set()}) == 1.0
    print("scorer ok")
