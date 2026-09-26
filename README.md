# Amazon ML Challenge 2026 — Business Entity Resolution

A clean, production-oriented repository for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Challenge Overview

The challenge resolves business entities across three heterogeneous data sources:
- **Source 1**: Canonical deduplicated reference source. Each Source 1 entity may link to zero, one, or multiple records from Source 2 and/or Source 3.
- **Source 2 & Source 3**: Independent noisy business records with no shared identifiers.
- **Fields**: `entity_id`, `business_name`, `business_address`, `country`.
- **Open-Set Country**: Training data contains US and India; test data additionally includes France. Country must be treated strictly as an open-set string field without hard-coding.

---

## 2. Repository Structure

```text
amazon-ml-challenge/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── preprocessing.py
│   ├── blocking.py
│   ├── features.py
│   ├── matching.py
│   └── evaluation.py
│
├── notebooks/
│   └── eda.ipynb
│
├── output/
│
├── config.yaml
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 3. Pipeline Architecture

```text
Raw Data (data/train/, data/test/)
   ↓
Data Validation
   ↓
EDA (notebooks/eda.ipynb)
   ↓
Preprocessing (src/preprocessing.py - preserves immutable raw fields)
   ↓
Blocking (src/blocking.py - scalable candidate generation)
   ↓
Pair Features (src/features.py)
   ↓
Matching Model (src/matching.py - parameter budget <= 8B)
   ↓
Decision Layer (src/matching.py - thresholding for singletons & multi-matches)
   ↓
Evaluation & Output (src/evaluation.py -> output/)
```

---

## 4. Official Output Requirements

The final pipeline generates two files in `output/`:
- `output/matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`)
- `output/candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`)

### Strict Submission Rules:
1. Every Source 1 test entity must appear exactly once.
2. If there are no matches (singleton): `matched_entity_ids` must be empty.
3. No duplicate IDs per row.
4. Only `S2-*` and `S3-*` IDs may appear as matches.
5. Every match in `matching_results.tsv` must exist in `candidate_pairs.tsv`.
6. `candidate_pairs.tsv` represents the **FINAL candidate set immediately before model scoring** (not an early high-volume blocking stage).

---

## 5. Evaluation Metric: Macro $F_{0.5}$

Evaluation uses entity-level, macro-averaged $F_{0.5}$:

$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

- **Precision Weighted**: Precision is weighted twice as heavily as recall ($\beta = 0.5$).
- **Singleton Handling**: Correctly predicting zero matches for a singleton receives full credit (1.0). Predicting an incorrect match receives zero (0.0).
- Computed independently per Source 1 entity and then macro-averaged.

---

## 6. Blocking Requirement

Amazon explicitly requires scalable candidate generation/blocking. Naive all-pairs comparison ($O(N^2)$) is strictly prohibited. `candidate_pairs.tsv` will be reviewed as part of final evaluation.

---

## 7. Fair Play & Licensing

- External data lookup is strictly prohibited (no external business lookup APIs, registries, geocoding, or internet augmentation).
- Total model parameter budget: $\le 8\text{B}$ parameters.
- Open-source licensing: MIT or Apache 2.0.

---

## 8. Setup Instructions

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt
```

---

## 9. Development Roadmap (TODOs)

- [ ] Place challenge datasets into `data/train/` and `data/test/`.
- [ ] Run `notebooks/eda.ipynb` for distribution analysis and hard negative identification.
- [ ] Implement data-driven normalization in `src/preprocessing.py` (preserving raw columns).
- [ ] Implement scalable indexing and candidate generation in `src/blocking.py`.
- [ ] Implement pairwise similarity features in `src/features.py`.
- [ ] Train pairwise matching model and calibrate precision threshold in `src/matching.py`.
- [ ] Evaluate with macro $F_{0.5}$ in `src/evaluation.py`.
- [ ] Export `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
