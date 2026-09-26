"""Data loading helpers.

TSVs are parsed once (sep="\\t", keep_default_na=False so empty strings stay
strings) and cached as parquet under cache/ for fast reloads.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset"
CACHE = ROOT / "cache"


def load(split: str, name: str) -> pd.DataFrame:
    """Load dataset/<split>/<split>_<name>.tsv as all-string columns, cached as parquet."""
    CACHE.mkdir(exist_ok=True)
    pq = CACHE / f"{split}_{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_csv(DATA / split / f"{split}_{name}.tsv", sep="\t", dtype=str,
                     keep_default_na=False, quoting=3)  # quoting=3: never treat quotes specially
    df.to_parquet(pq, index=False)
    return df


def load_sources(split: str):
    """Return (s1, s23) where s23 is Source 2 and Source 3 concatenated."""
    s1 = load(split, "source1")
    s23 = pd.concat([load(split, "source2"), load(split, "source3")], ignore_index=True)
    return s1, s23


def load_truth() -> dict:
    """Return {source1_entity_id: set(matched ids)} from the training ground truth."""
    gt = load("train", "ground_truth")
    return {k: set(v.split(",")) if v else set()
            for k, v in zip(gt.source1_entity_id, gt.matched_entity_ids)}
