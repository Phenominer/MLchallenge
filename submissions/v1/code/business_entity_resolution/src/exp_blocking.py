"""Blocking experiments on the 50k dev sample (train split): recall vs candidates per S1.

Compares forward top-k, reverse top-m and the address-key union.
Run: .venv/bin/python exp_blocking.py [fwd|rev|addr]
"""
import sys
import time

import numpy as np
import pandas as pd

from blocking import addr_block, blocking_recall, name_topk, reverse_topm
from evaluate import split_ids
from io_utils import CACHE, load_truth
from normalize import normalized


def main(mode):
    """Run one blocking experiment and print recall / avg candidates for several cutoffs."""
    truth = load_truth()
    s1 = normalized("train", "source1")
    pool = pd.concat([normalized("train", "source2"), normalized("train", "source3")], ignore_index=True)
    dev = s1[s1.entity_id.isin(set(split_ids()["dev"]))].reset_index(drop=True)
    t = time.time()
    if mode == "fwd":
        c = name_topk("train", dev, pool, 50)
        print("blocking done %.0fs" % (time.time() - t), flush=True)
        c.to_parquet(CACHE / "dev_fwd.parquet")
        for k in (5, 10, 15, 20, 30, 50):
            print("fwd k=%d recall=%.4f avg=%.1f" % (k, *blocking_recall(c[c.nrank <= k], dev, pool, truth)))
    elif mode == "rev":
        c = reverse_topm("train", s1, pool, 5)
        dev_rows = pd.Series(np.arange(len(dev)), index=dev.entity_id)
        c["q"] = dev_rows.reindex(s1.entity_id.values[c.q.values]).values
        c = c.dropna(subset=["q"]).astype({"q": int})
        c["rrank"] = c.groupby("p").nsim.rank(ascending=False, method="first")
        c.to_parquet(CACHE / "dev_rev.parquet")
        for m in (1, 2, 3, 5):
            print("rev m=%d recall=%.4f avg=%.1f" % (m, *blocking_recall(c[c.rrank <= m], dev, pool, truth)))
    elif mode == "addr":
        a = addr_block(dev, pool)
        print("addr only recall=%.4f avg=%.1f" % blocking_recall(a, dev, pool, truth))
        f = pd.read_parquet(CACHE / "dev_fwd.parquet")
        for k in (10, 20, 30):
            u = pd.concat([f[f.nrank <= k][["q", "p"]], a]).drop_duplicates()
            print("fwd k=%d + addr recall=%.4f avg=%.1f" % (k, *blocking_recall(u, dev, pool, truth)))
    print("time %.0fs" % (time.time() - t))


if __name__ == "__main__":
    main(sys.argv[1])
