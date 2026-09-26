"""Data preprocessing and normalization module.

Preserves immutable raw fields while generating normalized representations.
"""

from typing import Any, Dict, Optional
import pandas as pd


def normalize_business_name(raw_name: Optional[str]) -> str:
    """Normalizes a business name string.

    TODO:
    - Handle legal suffix variations (Inc, LLC, Ltd, Pvt Ltd, etc.)
    - Handle abbreviations and acronyms
    - Handle punctuation and whitespace variations
    - Handle typos, transliterations, and word-order permutations
    - Empirically validate against EDA distributions and hard negatives
    """
    if raw_name is None:
        return ""
    return str(raw_name).strip()


def normalize_business_address(raw_address: Optional[str]) -> str:
    """Normalizes a business address string.

    TODO:
    - Handle address abbreviations (St vs Street, Rd vs Road, Ave vs Avenue)
    - Handle landmark-based vs formal addresses
    - Handle municipal numbering and building/suite indicators
    - Handle missing components (postal code, city, state)
    - Preserve numbers crucial for distinguishing branches
    """
    if raw_address is None:
        return ""
    return str(raw_address).strip()


def preprocess_dataframe(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    is_train: bool = True,
) -> pd.DataFrame:
    """Preprocesses a source DataFrame while preserving raw fields.

    Raw columns (entity_id, business_name, business_address, country) must NOT be overwritten.
    Creates new normalized columns:
    - business_name_normalized
    - business_address_normalized
    """
    processed_df = df.copy()

    processed_df["business_name_normalized"] = processed_df["business_name"].apply(
        normalize_business_name
    )
    processed_df["business_address_normalized"] = processed_df["business_address"].apply(
        normalize_business_address
    )

    return processed_df
