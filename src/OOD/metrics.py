"""
Evaluation metrics for OOD detection.

OOD convention used throughout this module:

    ID  = 0
    OOD = 1

Higher OOD score means more OOD-like.

The main metric is AUROC.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .common import (
    validate_finite_array,
    validate_ood_scores,
)


def create_labels(
    num_id,
    num_ood,
):
    """
    Create binary labels.

    ID:
        0

    OOD:
        1
    """

    if num_id <= 0:
        raise ValueError(
            "num_id must be greater than zero."
        )

    if num_ood <= 0:
        raise ValueError(
            "num_ood must be greater than zero."
        )

    labels = np.concatenate(
        [
            np.zeros(num_id, dtype=np.int64),
            np.ones(num_ood, dtype=np.int64),
        ]
    )

    return labels


def combine_scores(
    id_scores,
    ood_scores,
):
    """
    Combine ID and OOD scores into one array.
    """

    id_scores = validate_ood_scores(
        id_scores
    )

    ood_scores = validate_ood_scores(
        ood_scores
    )

    scores = np.concatenate(
        [
            id_scores,
            ood_scores,
        ]
    )

    labels = create_labels(
        num_id=len(id_scores),
        num_ood=len(ood_scores),
    )

    return scores, labels


def calculate_auroc(
    id_scores,
    ood_scores,
):
    """
    Calculate AUROC.

    AUROC measures how well the OOD score separates:

        ID  -> lower scores
        OOD -> higher scores
    """

    scores, labels = combine_scores(
        id_scores,
        ood_scores,
    )

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Both ID and OOD samples are required."
        )

    return float(
        roc_auc_score(
            labels,
            scores,
        )
    )


def calculate_aupr(
    id_scores,
    ood_scores,
):
    """
    Calculate Area Under the Precision-Recall Curve.
    """

    scores, labels = combine_scores(
        id_scores,
        ood_scores,
    )

    return float(
        average_precision_score(
            labels,
            scores,
        )
    )


def calculate_roc_curve(
    id_scores,
    ood_scores,
):
    """
    Return ROC curve values.
    """

    scores, labels = combine_scores(
        id_scores,
        ood_scores,
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def calculate_detection_metrics(
    scores,
    labels,
    threshold,
):
    """
    Calculate classification metrics after applying
    an OOD threshold.

    Rule:

        score >= threshold
            -> OOD

        score < threshold
            -> ID
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    validate_finite_array(
        scores,
        name="scores",
    )

    if scores.ndim != 1:
        raise ValueError(
            "Scores must be 1-dimensional."
        )

    if labels.ndim != 1:
        raise ValueError(
            "Labels must be 1-dimensional."
        )

    if len(scores) != len(labels):
        raise ValueError(
            "Scores and labels must have the same length."
        )

    if not np.all(
        np.isin(
            labels,
            [0, 1],
        )
    ):
        raise ValueError(
            "Labels must contain only 0 and 1."
        )

    predictions = (
        scores >= threshold
    ).astype(np.int64)

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "num_samples": int(len(labels)),
        "num_id": int(np.sum(labels == 0)),
        "num_ood": int(np.sum(labels == 1)),
    }


def evaluate_method(
    id_scores,
    ood_scores,
    threshold=None,
):
    """
    Generate a complete evaluation summary.

    If threshold is supplied, threshold-based detection
    metrics are also returned.
    """

    auroc = calculate_auroc(
        id_scores,
        ood_scores,
    )

    aupr = calculate_aupr(
        id_scores,
        ood_scores,
    )

    result = {
        "auroc": auroc,
        "aupr": aupr,
        "id_samples": int(
            len(id_scores)
        ),
        "ood_samples": int(
            len(ood_scores)
        ),
    }

    if threshold is not None:
        scores, labels = combine_scores(
            id_scores,
            ood_scores,
        )

        detection = calculate_detection_metrics(
            scores=scores,
            labels=labels,
            threshold=threshold,
        )

        result.update(detection)

    return result


if __name__ == "__main__":
    print("=" * 70)
    print("MEDTRUST OOD METRICS")
    print("=" * 70)

    id_scores = np.array(
        [0.10, 0.20, 0.15, 0.30]
    )

    ood_scores = np.array(
        [0.80, 0.90, 0.70, 0.85]
    )

    print(
        "AUROC:",
        calculate_auroc(
            id_scores,
            ood_scores,
        ),
    )

    print(
        "AUPR:",
        calculate_aupr(
            id_scores,
            ood_scores,
        ),
    )

    scores, labels = combine_scores(
        id_scores,
        ood_scores,
    )

    print()
    print(
        calculate_detection_metrics(
            scores,
            labels,
            threshold=0.5,
        )
    )