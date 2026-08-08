"""
Calibration metrics for MedTrust.

This module contains reusable metrics for evaluating
probabilistic calibration of the 14-label chest X-ray
classifier.

All functions operate on NumPy arrays and are independent
of model inference and temperature fitting.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# =========================================================
# INPUT VALIDATION
# =========================================================

def _validate_binary_inputs(
    targets,
    probabilities,
):
    """
    Validate binary targets and predicted probabilities.

    Parameters
    ----------
    targets : array-like
        Binary ground-truth labels.

    probabilities : array-like
        Predicted probabilities in [0, 1].

    Returns
    -------
    targets, probabilities : np.ndarray
        Validated one-dimensional float64 arrays.
    """

    targets = np.asarray(
        targets,
        dtype=np.float64,
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if targets.shape != probabilities.shape:
        raise ValueError(
            "Targets and probabilities must have "
            "the same shape."
        )

    if targets.size == 0:
        raise ValueError(
            "Targets and probabilities cannot be empty."
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            "Non-finite values detected in targets."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite values detected in probabilities."
        )

    if not np.all(
        np.isin(targets, [0.0, 1.0])
    ):
        raise ValueError(
            "Targets must contain only binary values "
            "0 and 1."
        )

    if np.any(probabilities < 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError(
            "Probabilities must lie within [0, 1]."
        )

    return targets, probabilities


# =========================================================
# BINARY CROSS-ENTROPY / NEGATIVE LOG-LIKELIHOOD
# =========================================================

def binary_nll(
    targets,
    probabilities,
    eps=1e-12,
):
    """
    Calculate mean binary negative log-likelihood.

    This is equivalent to binary cross-entropy when
    evaluated on probabilities.
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    if eps <= 0.0 or eps >= 0.5:
        raise ValueError(
            "eps must satisfy 0 < eps < 0.5."
        )

    probabilities = np.clip(
        probabilities,
        eps,
        1.0 - eps,
    )

    losses = -(
        targets * np.log(probabilities)
        + (1.0 - targets)
        * np.log(1.0 - probabilities)
    )

    return float(
        np.mean(losses)
    )


# =========================================================
# BRIER SCORE
# =========================================================

def brier_score(
    targets,
    probabilities,
):
    """
    Calculate the mean Brier score.

    Lower values indicate better probabilistic predictions.
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    squared_errors = (
        probabilities - targets
    ) ** 2

    return float(
        np.mean(squared_errors)
    )


# =========================================================
# EXPECTED CALIBRATION ERROR
# =========================================================

def expected_calibration_error(
    targets,
    probabilities,
    n_bins=15,
):
    """
    Calculate Expected Calibration Error (ECE).

    Probabilities are divided into equal-width bins over
    [0, 1]. For every non-empty bin, the absolute difference
    between mean predicted probability and observed positive
    frequency is weighted by the fraction of samples in
    that bin.

    Parameters
    ----------
    targets : array-like
        Binary ground-truth labels.

    probabilities : array-like
        Predicted probabilities.

    n_bins : int
        Number of equal-width probability bins.

    Returns
    -------
    float
        Expected Calibration Error.
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, (bool, np.bool_))
        or n_bins <= 0
    ):
        raise ValueError(
            "n_bins must be a positive integer."
        )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    # digitize using interior boundaries gives:
    #
    # [0, edge_1)
    # [edge_1, edge_2)
    # ...
    # [edge_last, 1]
    #
    # Therefore probability exactly 1.0 remains inside
    # the final bin.
    bin_indices = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    ece = 0.0

    total_samples = targets.size

    for bin_idx in range(n_bins):

        mask = (
            bin_indices == bin_idx
        )

        bin_count = int(
            np.sum(mask)
        )

        if bin_count == 0:
            continue

        bin_probabilities = probabilities[
            mask
        ]

        bin_targets = targets[
            mask
        ]

        mean_confidence = float(
            np.mean(bin_probabilities)
        )

        observed_frequency = float(
            np.mean(bin_targets)
        )

        bin_weight = (
            bin_count / total_samples
        )

        ece += (
            bin_weight
            * abs(
                mean_confidence
                - observed_frequency
            )
        )

    return float(ece)


# =========================================================
# AUROC
# =========================================================

def binary_auroc(
    targets,
    probabilities,
):
    """
    Calculate binary AUROC.

    Returns NaN when the ground-truth targets contain only
    one class.
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    if len(np.unique(targets)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            targets,
            probabilities,
        )
    )


# =========================================================
# COMBINED CALIBRATION METRICS
# =========================================================

def calculate_binary_calibration_metrics(
    targets,
    probabilities,
    n_bins=15,
):
    """
    Calculate the complete calibration metric set for
    one disease.
    """

    targets, probabilities = _validate_binary_inputs(
        targets,
        probabilities,
    )

    return {
        "nll": binary_nll(
            targets,
            probabilities,
        ),
        "brier": brier_score(
            targets,
            probabilities,
        ),
        "ece": expected_calibration_error(
            targets,
            probabilities,
            n_bins=n_bins,
        ),
        "auroc": binary_auroc(
            targets,
            probabilities,
        ),
    }