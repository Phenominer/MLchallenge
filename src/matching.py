"""Matching model and decision layer module.

Supports:
- Zero matches (singletons) -> matched_entity_ids is empty
- Single match -> single entity ID
- Multiple matches -> delimited entity IDs
- Precision-oriented decision thresholding
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseMatchingModel(ABC):
    """Abstract base class for pairwise matching models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseMatchingModel":
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        pass


class BaselineMatchingModel(BaseMatchingModel):
    """Minimal scaffolding matching model."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaselineMatchingModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.array([], dtype=float)
        return np.zeros(len(X), dtype=float)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


def score_candidates(
    candidate_pairs_df: pd.DataFrame,
    features_df: pd.DataFrame,
    model: BaseMatchingModel,
) -> pd.DataFrame:
    """Scores candidate pairs using matching model probabilities."""
    if len(candidate_pairs_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "match_score"])

    scored_df = candidate_pairs_df.copy()
    scored_df["match_score"] = model.predict_proba(features_df)
    return scored_df


def select_matches(
    scored_candidates_df: pd.DataFrame,
    all_source1_ids: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Selects matches above threshold for each Source 1 entity.

    Rules:
    - Every Source 1 entity must appear exactly once.
    - If no matches: matched_entity_ids is empty string.
    - Only S2-* and S3-* IDs are allowed.
    - No duplicate IDs within a row.
    """
    if scored_candidates_df.empty or "match_score" not in scored_candidates_df.columns:
        return pd.DataFrame(
            [{"source1_entity_id": s1_id, "matched_entity_ids": ""} for s1_id in all_source1_ids],
            columns=["source1_entity_id", "matched_entity_ids"],
        )

    accepted = scored_candidates_df[scored_candidates_df["match_score"] >= threshold]
    matches_by_s1 = accepted.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: " ".join(sorted(set(str(i) for i in ids if str(i).startswith(("S2-", "S3-")))))
    ).to_dict()

    results = []
    for s1_id in all_source1_ids:
        results.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matches_by_s1.get(s1_id, ""),
        })

    return pd.DataFrame(results, columns=["source1_entity_id", "matched_entity_ids"])
