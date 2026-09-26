"""Pairwise feature engineering module."""

from typing import Any, Dict, Optional
import pandas as pd


def extract_pair_features(
    source1_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Extracts pairwise similarity features between Source 1 and candidate entities.

    TODO:
    - Name exact match (binary flag on raw and normalized strings)
    - Normalized name similarity (Levenshtein, Jaro-Winkler)
    - Character similarity (n-gram Jaccard / Dice)
    - Token similarity (TF-IDF cosine, token overlap)
    - Address similarity (token Jaccard, fuzzy match)
    - Country match (open-set categorical comparison)
    - Numerical / address component matches (building number, postal code)
    - Source origin indicator (Source 2 vs Source 3)

    Returns:
        pd.DataFrame containing feature matrix with pair identification columns:
        ['source1_entity_id', 'candidate_entity_id', ...]
    """
    return pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "name_exact_match",
            "name_similarity_levenshtein",
            "address_similarity_token",
            "country_match",
        ]
    )
