"""
OOD threshold selection.

The threshold is selected using the validation set only.

Important data separation:

    Training:
        Used for Mahalanobis statistics.

    Validation:
        Used for threshold selection.

    Test:
        Used only for final evaluation.

Default method:

    Youden's J statistic

        J = TPR - FPR

The threshold producing the maximum J is selected.
"""

import numpy as np
from sklearn.metrics import roc_curve

from .common import (
    ensure_dir,
    save_json,
    validate_finite_array,
)
from .config import (
    ARTIFACT_OUTPUT_DIR,
    THRESHOLD_METHOD,
)
from .metrics import combine_scores


THRESHOLD_OUTPUT_PATH = (
    ARTIFACT_OUTPUT_DIR
    / "ood_thresholds.json"
)


def select_threshold_youden(
    id_scores,
    ood_scores,
):
    """
    Select threshold using Youden's J statistic.

    J = TPR - FPR

    The threshold maximizing J is selected.
    """

    scores, labels = combine_scores(
        id_scores=id_scores,
        ood_scores=ood_scores,
    )

    validate_finite_array(
        scores,
        name="validation scores",
    )

    fpr, tpr, thresholds = roc_curve(
        labels,
        scores,
    )

    # sklearn can return infinity as the first threshold.
    finite = np.isfinite(thresholds)

    if not np.any(finite):
        raise RuntimeError(
            "No finite ROC threshold was found."
        )

    valid_fpr = fpr[finite]
    valid_tpr = tpr[finite]
    valid_thresholds = thresholds[finite]

    youden_j = (
        valid_tpr
        - valid_fpr
    )

    best_index = int(
        np.argmax(youden_j)
    )

    threshold = float(
        valid_thresholds[best_index]
    )

    return {
        "threshold": threshold,
        "youden_j": float(
            youden_j[best_index]
        ),
        "tpr": float(
            valid_tpr[best_index]
        ),
        "fpr": float(
            valid_fpr[best_index]
        ),
        "method": "youden_j",
    }


def select_threshold_id_percentile(
    id_scores,
    target_false_positive_rate=0.10,
):
    """
    Select an OOD threshold using ID validation scores only.

    Higher scores indicate more OOD-like inputs.

    The threshold is the (1 - target FPR) percentile.
    With the rule score >= threshold -> OOD, the observed
    validation false-positive rate is reported explicitly.
    Ties can cause the observed rate to exceed the target.
    """
    scores = np.asarray(
        id_scores,
        dtype=np.float64,
    )

    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            "ID validation scores must be a non-empty 1-D array."
        )

    if not 0.0 < target_false_positive_rate < 1.0:
        raise ValueError(
            "target_false_positive_rate must be between 0 and 1."
        )

    validate_finite_array(
        scores,
        name="ID validation scores",
    )

    percentile = 1.0 - target_false_positive_rate

    # Use the higher observed score to avoid interpolating
    # a threshold between two actual validation scores.
    try:
        threshold = np.quantile(
            scores,
            percentile,
            method="higher",
        )
    except TypeError:
        # Compatibility with older NumPy versions.
        threshold = np.quantile(
            scores,
            percentile,
            interpolation="higher",
        )

    observed_fpr = float(
        np.mean(scores >= threshold)
    )

    return {
        "threshold": float(threshold),
        "method": "id_percentile",
        "percentile": float(percentile * 100.0),
        "target_false_positive_rate": float(
            target_false_positive_rate
        ),
        "observed_false_positive_rate": observed_fpr,
        "validation_samples": int(scores.size),
    }

def select_threshold(
    id_scores,
    ood_scores,
    method=THRESHOLD_METHOD,
):
    """
    Select an OOD threshold.

    Currently supported:

        youden_j
    """

    if method == "youden_j":
        return select_threshold_youden(
            id_scores=id_scores,
            ood_scores=ood_scores,
        )

    raise ValueError(
        f"Unsupported threshold method: {method}"
    )


def select_thresholds_for_methods(
    validation_scores,
):
    """
    Select thresholds for multiple OOD methods.

    Expected format:

        {
            "msp": {
                "id": [...],
                "ood": [...]
            },

            "energy": {
                "id": [...],
                "ood": [...]
            },

            "mahalanobis": {
                "id": [...],
                "ood": [...]
            }
        }
    """

    results = {}

    for method, values in validation_scores.items():

        if "id" not in values:
            raise KeyError(
                f"Missing ID scores for method: {method}"
            )

        if "ood" not in values:
            raise KeyError(
                f"Missing OOD scores for method: {method}"
            )

        results[method] = select_threshold(
            id_scores=np.asarray(
                values["id"],
                dtype=np.float64,
            ),
            ood_scores=np.asarray(
                values["ood"],
                dtype=np.float64,
            ),
        )

    return results



def save_thresholds(
    thresholds,
    output_path=THRESHOLD_OUTPUT_PATH,
    threshold_method=None,
):
    """Save selected validation thresholds."""

    ensure_dir(output_path.parent)

    payload = {
        "threshold_method": (
            threshold_method or THRESHOLD_METHOD
        ),
        "source": "validation_only",
        "calibration_dataset": "NIH ChestX-ray14 ID validation",
        "target_false_positive_rate": 0.10,
        "test_data_used": False,
        "ood_validation_data_used": False,
        "thresholds": thresholds,
    }

    save_json(payload, output_path)

    print(f"Saved thresholds: {output_path}")


def load_thresholds(
    input_path=THRESHOLD_OUTPUT_PATH,
):
    """
    Load previously selected thresholds.
    """

    if not input_path.exists():
        raise FileNotFoundError(
            "OOD threshold file not found:\n"
            f"{input_path}"
        )

    import json

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if "thresholds" not in data:
        raise RuntimeError(
            "Threshold file does not contain 'thresholds'."
        )

    return data


if __name__ == "__main__":
    print("=" * 70)
    print("MEDTRUST OOD THRESHOLD SELECTION")
    print("=" * 70)

    id_scores = np.array(
        [0.10, 0.20, 0.15, 0.25, 0.30]
    )

    ood_scores = np.array(
        [0.70, 0.80, 0.90, 0.85, 0.75]
    )

    result = select_threshold_youden(
        id_scores=id_scores,
        ood_scores=ood_scores,
    )

    print()
    print("Selected threshold:")
    print(result)

    print()
    print(
        "Rule: score >= threshold -> OOD"
    )