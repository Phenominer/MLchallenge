"""End-to-end pipeline: blocking -> pair features -> LightGBM -> one-to-one -> thresholds.

Modes:
  dev   train on a sample of the train split, tune thresholds on the 50k dev sample
  val   score the saved model on the full 441k validation split
  test  run the saved model on the test set and write output/*.tsv
"""
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import addr_block, blocking_recall, name_topk
from evaluate import SEED, f05_macro, split_ids
from features import pair_features
from io_utils import CACHE, ROOT, load_truth
from normalize import normalized

K = 20              # name top-k per S1
ADDR_MAX_FREQ = 30  # address-key block size cap
N_TRAIN = 100_000   # S1 entities used to train the matcher
DROP = ["p_ncand", "p_gap"]  # depend on how many S1 are queried -> train/test skew


def log(*a):
    """Print with a timestamp."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def candidates(tag: str, name: str, q: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Union of name top-K and address-key candidates for query frame q (cached by name)."""
    path = CACHE / f"cand_{name}_k{K}_a{ADDR_MAX_FREQ}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    c = name_topk(tag, q, pool, K)
    a = addr_block(q, pool, ADDR_MAX_FREQ)
    c = c.merge(a, on=["q", "p"], how="outer")
    c = c.sort_values(["q", "p"], ignore_index=True)
    c.to_parquet(path)
    return c


def labels(q, pool, cand, truth):
    """1 if (S1, pool) pair is a true match."""
    pairs = set((k, m) for k in q.entity_id for m in truth[k])
    return np.fromiter((x in pairs for x in zip(q.entity_id.values[cand.q.values],
                                                 pool.entity_id.values[cand.p.values])), bool, len(cand))


def decide(cand: pd.DataFrame, prob: np.ndarray, t_top: float, t_rest: float) -> pd.DataFrame:
    """One-to-one (each pool row -> its best S1), then keep the S1's best pair if prob >= t_top
    and any other pair if prob >= t_rest. Returns the kept (q, p) rows."""
    d = cand[["q", "p"]].assign(prob=prob)
    d = d.loc[d.groupby("p").prob.idxmax()]
    d["r"] = d.groupby("q").prob.rank(ascending=False, method="first")
    return d[((d.r == 1) & (d.prob >= t_top)) | (d.prob >= t_rest)]


def tune(cand, prob, q, pool, truth):
    """Grid-search (t_top, t_rest) maximizing macro F0.5 on the given query set."""
    best = (-1, None)
    for t_top in np.arange(0.1, 0.9, 0.05):
        for t_rest in np.arange(t_top, 0.95, 0.05):
            s = score(decide(cand, prob, t_top, t_rest), q, pool, truth)
            if s > best[0]:
                best = (s, (round(t_top, 2), round(t_rest, 2)))
    return best


def score(kept, q, pool, truth):
    """Macro F0.5 of kept (q, p) rows over all entities in q."""
    pred = {}
    for e, m in zip(q.entity_id.values[kept.q.values], pool.entity_id.values[kept.p.values]):
        pred.setdefault(e, set()).add(m)
    return f05_macro(pred, {e: truth[e] for e in q.entity_id})


def load_all(split):
    """Normalized S1 table and concatenated S2+S3 pool for a split."""
    s1 = normalized(split, "source1")
    pool = pd.concat([normalized(split, "source2"), normalized(split, "source3")], ignore_index=True)
    return s1, pool


def run_dev():
    """Train on N_TRAIN train-split entities, tune thresholds on dev, save model."""
    truth = load_truth()
    s1, pool = load_all("train")
    ids = split_ids()
    rng = np.random.default_rng(SEED)
    tr = s1[s1.entity_id.isin(set(rng.choice(ids["train"], N_TRAIN, replace=False)))].reset_index(drop=True)
    dev = s1[s1.entity_id.isin(set(ids["dev"]))].reset_index(drop=True)
    sets = {}
    for nm, q in (("trs", tr), ("dev", dev)):
        c = candidates("train", nm, q, pool)
        r, avg = blocking_recall(c, q, pool, truth)
        log(f"{nm}: blocking recall={r:.4f} avg cand/S1={avg:.1f}")
        X = pair_features(q, pool, c)
        y = labels(q, pool, c, truth)
        sets[nm] = (q, c, X.drop(columns=DROP), y)
        log(f"{nm}: {len(c)} pairs, pos rate {y.mean():.3f}")
    _, _, Xt, yt = sets["trs"]
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=127, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED,
                               verbose=-1)
    model.fit(Xt, yt)
    q, c, Xd, yd = sets["dev"]
    prob = model.predict_proba(Xd)[:, 1]
    s, th = tune(c, prob, q, pool, truth)
    log(f"dev F0.5={s:.4f} thresholds(top,rest)={th}")
    log("dev F0.5 @0.5/0.5 =", round(score(decide(c, prob, 0.5, 0.5), q, pool, truth), 4))
    imp = pd.Series(model.feature_importances_, Xd.columns).sort_values(ascending=False)
    log("importance:", imp.head(15).to_dict())
    pickle.dump({"model": model, "th": th, "cols": list(Xd.columns)}, open(CACHE / "model.pkl", "wb"))


def predict(split, name, q, pool):
    """Candidates + model probabilities + decision for query frame q."""
    m = pickle.load(open(CACHE / "model.pkl", "rb"))
    c = candidates(split, name, q, pool)
    X = pair_features(q, pool, c)[m["cols"]]
    prob = m["model"].predict_proba(X)[:, 1]
    return c, decide(c, prob, *m["th"])


def run_val():
    """Score the saved model on the full validation split."""
    truth = load_truth()
    s1, pool = load_all("train")
    q = s1[s1.entity_id.isin(set(split_ids()["val"]))].reset_index(drop=True)
    c, kept = predict("train", "val", q, pool)
    r, avg = blocking_recall(c, q, pool, truth)
    log(f"val: blocking recall={r:.4f} avg cand/S1={avg:.1f} F0.5={score(kept, q, pool, truth):.4f}")


def write_lists(path: Path, col: str, q, pool, rows):
    """Write one row per S1 entity with a comma-joined, de-duplicated list of pool ids."""
    ids = pd.Series(pool.entity_id.values[rows.p.values]).groupby(q.entity_id.values[rows.q.values]).agg(
        lambda s: ",".join(dict.fromkeys(s)))
    out = pd.DataFrame({"source1_entity_id": q.entity_id.values})
    out[col] = out.source1_entity_id.map(ids).fillna("")
    out.to_csv(path, sep="\t", index=False, quoting=3)


def run_test():
    """Run on test and write output/matching_results.tsv and output/candidate_pairs.tsv."""
    q, pool = load_all("test")
    c, kept = predict("test", "test", q, pool)
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    write_lists(out / "candidate_pairs.tsv", "candidate_entity_ids", q, pool, c)
    write_lists(out / "matching_results.tsv", "matched_entity_ids", q, pool, kept)
    log(f"test: avg cand/S1={len(c) / len(q):.1f} avg matches/S1={len(kept) / len(q):.2f} "
        f"empty share={1 - kept.q.nunique() / len(q):.3f}")


if __name__ == "__main__":
    {"dev": run_dev, "val": run_val, "test": run_test}[sys.argv[1]]()
