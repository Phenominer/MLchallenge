"""Official evaluation metrics: entity-level and macro-averaged F_0.5.

F_0.5 Formula:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Official Scoring Rules:
- F_0.5 is calculated independently for each Source 1 entity and macro-averaged.
- Precision is weighted twice as heavily as recall (beta = 0.5).
- Correctly predicting a singleton (zero matches) receives full credit (1.0).
- Predicting an incorrect match for a singleton receives zero credit (0.0).
"""

from typing import Dict, List, Set, Union
import numpy as np


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Computes F_beta score given precision and recall."""
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_entity_f05(
    predicted_ids: Union[Set[str], List[str]],
    ground_truth_ids: Union[Set[str], List[str]],
) -> float:
    """Computes F_0.5 score for an individual Source 1 entity.

    - If Ground Truth is empty (singleton):
        - Pred empty -> 1.0 (full credit)
        - Pred non-empty -> 0.0
    - If Ground Truth is non-empty:
        - Pred empty -> 0.0
        - Otherwise -> compute Precision, Recall, and F_0.5
    """
    pred_set = {str(x).strip() for x in predicted_ids if str(x).strip()}
    gt_set = {str(x).strip() for x in ground_truth_ids if str(x).strip()}

    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    if len(pred_set) == 0:
        return 0.0

    true_positives = len(pred_set.intersection(gt_set))
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(pred_set)
    recall = true_positives / len(gt_set)

    return compute_f_beta(precision, recall, beta=0.5)


def compute_macro_f05(
    predictions: Dict[str, Union[Set[str], List[str]]],
    ground_truth: Dict[str, Union[Set[str], List[str]]],
) -> float:
    """Computes macro-averaged F_0.5 score across all Source 1 entities."""
    if not ground_truth:
        return 0.0

    scores: List[float] = []
    for s1_id, gt_ids in ground_truth.items():
        pred_ids = predictions.get(s1_id, set())
        scores.append(compute_entity_f05(pred_ids, gt_ids))

    return float(np.mean(scores))
