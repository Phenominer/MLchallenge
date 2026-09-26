# Business Entity Resolution (Amazon ML Challenge 2026)

Blocking (name + address TF-IDF top-k, address key) -> pair features (rapidfuzz, set overlaps)
-> LightGBM matcher -> one-to-one assignment -> F0.5-tuned thresholds.
No external data, no LLMs; all libraries are BSD/MIT/Apache licensed.

## Setup

Run from the repository root (the folder that contains `dataset/` and `utils/`).

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
```

## Reproduce end to end

```bash
cd code/business_entity_resolution/src
PY=../../../.venv/bin/python

# 1. mine the Indic -> Latin token dictionary from training pairs (cache/indic_dict.json)
$PY -c "from translit import build_dictionary; build_dictionary('../../../cache/indic_dict.json')"

# 2. train the matcher on a train-split sample, tune thresholds on the dev sample (cache/model.pkl)
$PY pipeline.py dev

# 3. (optional) score on the full 441k validation split
$PY pipeline.py val

# 4. predict on test, writes output/matching_results.tsv and output/candidate_pairs.tsv
$PY pipeline.py test

# 5. validate
cd ../../.. && python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Normalized tables, TF-IDF indexes and candidate sets are cached in `cache/`; delete it to
recompute from scratch. Optional analysis scripts: `eda.py` (step 1 statistics) and
`exp_blocking.py fwd|rev|addr` (blocking experiments on the dev sample).

## Modules (`src/`)

| File | Purpose |
|---|---|
| `io_utils.py` | TSV loading (`sep="\t"`, `keep_default_na=False`), parquet cache |
| `translit.py` | Indic script transliteration + dictionary mined from training pairs |
| `normalize.py` | Name/address normalization, legal suffix field, numbers, address key |
| `evaluate.py` | 80/20 split by S1 entity (seed 42) and the official macro F0.5 scorer |
| `blocking.py` | Per-country TF-IDF sparse top-k and address-key blocking |
| `features.py` | Pair features |
| `pipeline.py` | Training, threshold tuning, one-to-one rule, output writing |
