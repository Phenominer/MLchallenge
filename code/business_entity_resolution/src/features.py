"""Pair features for candidate (S1 row q, pool row p) pairs.

String similarities use rapidfuzz.process.cpdist (element-wise, C++); set-based
features run in a process pool. Country is NOT a feature (must generalize to France).
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

SCORERS = {"jw": JaroWinkler.normalized_similarity, "tset": fuzz.token_set_ratio,
           "tsort": fuzz.token_sort_ratio, "part": fuzz.partial_ratio, "ratio": fuzz.ratio}


def _jacc(a: str, b: str) -> float:
    """Token Jaccard of two space-separated strings (NaN when both empty)."""
    a, b = set(a.split()), set(b.split())
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def _set_feats(args):
    """Set-based features for one chunk of aligned string columns."""
    qn, pn, qa, pa, qnum, pnum, ql, pl = args
    out = np.empty((len(qn), 7), np.float32)
    for i in range(len(qn)):
        a, b = qnum[i].split(), pnum[i].split()
        sa, sb = set(a), set(b)
        out[i] = (_jacc(qn[i], pn[i]), _jacc(qa[i], pa[i]),
                  len(sa & sb) / len(sa | sb) if sa | sb else np.nan,
                  (a[0] == b[0]) if a and b else np.nan,          # first (street) number equal
                  max((len(x) for x in sa & sb if len(x) >= 5), default=0) > 0,  # shared postal-like number
                  (ql[i] == pl[i]) if ql[i] and pl[i] else np.nan,  # legal suffix agreement
                  _jacc(ql[i], pl[i]))
    return out


def pair_features(q: pd.DataFrame, pool: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Return a float32 feature frame aligned with cand (columns q, p, optional nsim/nrank)."""
    A, B = q.iloc[cand.q.values], pool.iloc[cand.p.values]
    f = {}
    for fld in ("name", "addr"):
        a, b = A[fld].tolist(), B[fld].tolist()
        for nm, sc in SCORERS.items():
            f[f"{fld}_{nm}"] = process.cpdist(a, b, scorer=sc, workers=-1).astype(np.float32)
    cols = [A.name.values, B.name.values, A.addr.values, B.addr.values,
            A.nums.values, B.nums.values, A.legal.values, B.legal.values]
    n, step = len(cand), 250_000
    with Pool(10) as pl:
        s = np.vstack(pl.map(_set_feats, [[c[i:i + step] for c in cols] for i in range(0, n, step)]))
    for j, nm in enumerate(["name_jacc", "addr_jacc", "num_jacc", "num_first", "postal", "legal_eq", "legal_jacc"]):
        f[nm] = s[:, j]
    f["akey_eq"] = ((A.akey.values == B.akey.values) & (A.akey.values != "")).astype(np.float32)
    f["p_addr_empty"] = (B.addr.values == "").astype(np.float32)
    f["p_is_dom"] = B.is_dom.values.astype(np.float32)
    f["p_src3"] = (B.entity_id.str[:2].values == "S3").astype(np.float32)
    f["len_ratio"] = (B.name.str.len().values + 1) / (A.name.str.len().values + 1)
    f["nsim"] = cand.nsim.values if "nsim" in cand else np.nan
    df = pd.DataFrame(f).astype(np.float32)
    # group-relative features: how this pair compares with the other candidates of the same S1 / pool row
    df["score0"] = df.name_tset + df.addr_tset
    g = df.score0.groupby(cand.q.values)
    df["q_ncand"] = g.transform("size").values
    df["q_rank"] = g.rank(ascending=False, method="min").values
    df["q_gap"] = (g.transform("max") - df.score0).values
    gp = df.score0.groupby(cand.p.values)
    df["p_ncand"] = gp.transform("size").values
    df["p_gap"] = (gp.transform("max") - df.score0).values
    return df
