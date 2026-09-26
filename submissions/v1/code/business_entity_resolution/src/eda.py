"""Step 1: data exploration. Prints counts, match distribution, sample pairs.

Run: .venv/bin/python code/business_entity_resolution/src/eda.py
"""
from collections import Counter

import pandas as pd

from io_utils import load_sources, load_truth


def main():
    """Print dataset statistics and ~25 true matched pairs side by side."""
    for split in ("train", "test"):
        s1, s23 = load_sources(split)
        s23 = s23.assign(src=s23.entity_id.str[:2])
        print(f"\n=== {split} ===")
        print("S1 by country:\n", s1.country.value_counts().to_string())
        print("S2/S3 by country:\n", s23.groupby(["src", "country"]).size().to_string())
        print("empty address share S1/S23:",
              round((s1.business_address == "").mean(), 3), round((s23.business_address == "").mean(), 3))

    s1, s23 = load_sources("train")
    truth = load_truth()
    sizes = pd.Series({k: len(v) for k, v in truth.items()})
    print("\nGT rows:", len(truth), " S1 rows:", len(s1))
    print("singleton share:", round((sizes == 0).mean(), 4))
    print("matches per S1:\n", sizes.clip(upper=10).value_counts().sort_index().to_string())
    by_c = s1.set_index("entity_id").country
    print("singleton share by country:\n", (sizes == 0).groupby(by_c.reindex(sizes.index)).mean().to_string())
    src = Counter(m[:2] for v in truth.values() for m in v)
    print("match share S2 vs S3:", {k: round(n / sum(src.values()), 3) for k, n in src.items()})
    all_m = [m for v in truth.values() for m in v]
    print("matched ids total / unique (does an S2/S3 record match >1 S1?):", len(all_m), len(set(all_m)))
    print("share of S2/S3 pool that is matched to some S1:", round(len(set(all_m)) / len(s23), 3))

    # Floor: all-empty prediction scores 1.0 on singletons, 0.0 elsewhere.
    print("ALL-EMPTY F0.5 floor (train):", round((sizes == 0).mean(), 4))

    rec = pd.concat([s1, s23]).set_index("entity_id")
    sample = [(k, m) for k, v in list(truth.items())[:400] for m in sorted(v)]
    for k, m in pd.Series(sample).sample(25, random_state=0):
        a, b = rec.loc[k], rec.loc[m]
        print(f"\n[{a.country}] {a.business_name} | {a.business_address}\n"
              f"  {m[:2]} {b.business_name} | {b.business_address}")


if __name__ == "__main__":
    main()
