"""Candidate generation (blocking), always within the same country string.

1. TF-IDF blocking: one vector per record = [name char_wb 3-4 gram TF-IDF,
   address word TF-IDF], each L2-normalized and weighted (cosine = weighted mix of
   name and address cosine). Very frequent features are dropped (max_df), which
   keeps sparse top-k fast. sparse_dot_topn, never a dense/cartesian comparison.
   Direction 'fwd': each S1 query gets its top-k pool records.
   Direction 'rev': each pool record gets its top-m S1 records, inverted per S1.
2. Address blocking: exact match on the address key (street number + first 4 chars
   of the street name); keys too frequent in the pool are dropped.
Candidates are returned as integer row indices (q = S1 row, p = pool row).
"""
import pickle
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from io_utils import CACHE


def _topk(A, B, k):
    """Return (row, col, score) of the top-k columns of A @ B.T per row, chunked to bound memory."""
    rows, cols, vals = [], [], []
    Bt = B.T.tocsr()
    for s in range(0, A.shape[0], 200_000):
        C = sp_matmul_topn(A[s:s + 200_000], Bt, top_n=k, threshold=0.05, n_threads=10).tocoo()
        rows.append(C.row + s), cols.append(C.col), vals.append(C.data)
        print(f"    topk rows {min(s + 200_000, A.shape[0])}/{A.shape[0]}", flush=True)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


W_NAME = 0.5  # weight of name cosine in the blocking score (address gets 1 - W_NAME)


class RecordVectorizer:
    """Name char n-gram TF-IDF + address word TF-IDF, stacked with sqrt weights."""

    def __init__(self):
        """Create the two sub-vectorizers (max_df drops features too common to discriminate; they dominate matmul cost)."""
        kw = dict(min_df=2, dtype=np.float32, sublinear_tf=True)
        # caps tuned on dev: 4.4x faster top-k for -1.6 pt recall@20 versus no cap
        self.nv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_df=0.0033, **kw)
        self.av = TfidfVectorizer(token_pattern=r"\w+", max_df=0.01, **kw)

    def _stack(self, N, A):
        """Weighted hstack so that dot product = W*name_cos + (1-W)*addr_cos."""
        return sp.hstack([N * np.float32(W_NAME ** 0.5), A * np.float32((1 - W_NAME) ** 0.5)], format="csr")

    def fit_transform(self, df):
        """Fit on a pool frame and return its matrix."""
        return self._stack(self.nv.fit_transform(df.name.values), self.av.fit_transform(df.addr.values))

    def transform(self, df):
        """Vectorize query records."""
        return self._stack(self.nv.transform(df.name.values), self.av.transform(df.addr.values))


def country_index(tag: str, pool: pd.DataFrame, country: str):
    """Fit (or load cached) record TF-IDF on one country's pool. Returns (vectorizer, matrix, row ids)."""
    path = CACHE / f"tfidf5_{tag}_{country}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    rows = np.flatnonzero(pool.country.values == country)
    vec = RecordVectorizer()
    M = vec.fit_transform(pool.iloc[rows])
    out = (vec, M, rows)
    pickle.dump(out, open(path, "wb"), protocol=4)
    return out


def name_topk(tag: str, q: pd.DataFrame, pool: pd.DataFrame, k: int) -> pd.DataFrame:
    """Forward TF-IDF blocking: top-k pool rows per query row, per country. Returns q, p, nsim, nrank."""
    out = []
    for c in q.country.unique():  # open set: every query country, seen or not
        qr = np.flatnonzero(q.country.values == c)
        if not (pool.country.values == c).any():
            continue
        t = time.time()
        vec, M, pr = country_index(tag, pool, c)
        t1 = time.time()
        r, col, v = _topk(vec.transform(q.iloc[qr]), M, k)
        print(f"  {c}: {len(qr)} queries, index {t1 - t:.0f}s, topk {time.time() - t1:.0f}s", flush=True)
        out.append(pd.DataFrame({"q": qr[r], "p": pr[col], "nsim": v}))
    df = pd.concat(out, ignore_index=True)
    df["nrank"] = df.groupby("q").nsim.rank(ascending=False, method="first").astype(np.int16)
    return df


def reverse_topm(tag: str, s1_all: pd.DataFrame, pool: pd.DataFrame, m: int) -> pd.DataFrame:
    """Reverse TF-IDF blocking: top-m S1 rows (of the full S1 table) per pool row. Returns q, p, nsim."""
    out = []
    for c in s1_all.country.unique():
        qr = np.flatnonzero(s1_all.country.values == c)
        if not (pool.country.values == c).any():
            continue
        vec, M, pr = country_index(tag, pool, c)
        r, col, v = _topk(M, vec.transform(s1_all.iloc[qr]), m)
        out.append(pd.DataFrame({"q": qr[col], "p": pr[r], "nsim": v}))
    return pd.concat(out, ignore_index=True)


def addr_block(q: pd.DataFrame, pool: pd.DataFrame, max_freq: int = 30) -> pd.DataFrame:
    """Exact address-key join within country; skip keys shared by more than max_freq pool rows."""
    pk = pd.DataFrame({"key": pool.country.values + "|" + pool.akey.values, "p": np.arange(len(pool))})
    pk = pk[pool.akey.values != ""]
    freq = pk.key.map(pk.key.value_counts())
    pk = pk[freq <= max_freq]
    qk = pd.DataFrame({"key": q.country.values + "|" + q.akey.values, "q": np.arange(len(q))})[q.akey.values != ""]
    return qk.merge(pk, on="key")[["q", "p"]]


def blocking_recall(cand: pd.DataFrame, q: pd.DataFrame, pool: pd.DataFrame, truth: dict):
    """Return (pair recall, avg candidates per S1) for candidate rows (q, p) against truth."""
    pairs = set(zip(q.entity_id.values[cand.q.values], pool.entity_id.values[cand.p.values]))
    n_true = sum(len(truth[e]) for e in q.entity_id)
    hit = sum(1 for e in q.entity_id for m in truth[e] if (e, m) in pairs)
    return hit / n_true, len(cand) / len(q)
