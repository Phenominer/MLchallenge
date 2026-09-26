"""Scalable candidate generation and blocking module.

Candidate pairs represent the FINAL candidate set immediately before
the matching model scores candidates. Naive all-pairs comparison is strictly prohibited.
"""

from typing import Any, Dict, Optional
import pandas as pd


def generate_candidates(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generates candidate pairs mapping each Source 1 entity to candidate S2/S3 entities.

    TODO:
    - Exact normalized key blocking on name and address keys
    - Token-based inverted indexing (BM25 / TF-IDF)
    - Character and word n-gram blocking for typo tolerance
    - Address / locality-based blocking
    - Multi-block union to combine candidates across orthogonal keys
    - Candidate recall evaluation: |TrueMatches ∩ Candidates| / |TrueMatches|
    - Candidate pruning to top-K candidates per Source 1 entity

    Returns:
        pd.DataFrame with columns ['source1_entity_id', 'candidate_entity_ids'].
        Every Source 1 entity must appear exactly once.
    """
    # Scaffolding placeholder: ensures every Source 1 entity is represented
    candidates = [
        {"source1_entity_id": s1_id, "candidate_entity_ids": ""}
        for s1_id in source1_df["entity_id"]
    ]
    return pd.DataFrame(candidates, columns=["source1_entity_id", "candidate_entity_ids"])
