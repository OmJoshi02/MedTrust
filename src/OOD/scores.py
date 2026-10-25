"""
OOD scoring methods for MedTrust.

Implemented methods:

1. MSP-like score
2. Energy score
3. Mahalanobis score

For all methods:

    higher score = more OOD-like

The classifier itself remains frozen.
"""

import numpy as np

from .common import validate_finite_array, validate_ood_scores
from .config import ENERGY_TEMPERATURE


def msp_score(logits):
    """
    Calculate an MSP-style OOD score for multi-label classification.

    Since MedTrust is a multi-label classifier, sigmoid probabilities
    are used instead of softmax probabilities.

    Formula:

        confidence = max(sigmoid(logits))

        OOD score = 1 - confidence

    Therefore:

        low score  -> more ID-like
        high score -> more OOD-like
    """

    logits = np.asarray(logits, dtype=np.float64)

    if logits.ndim != 2:
        raise ValueError(
            "Logits must have shape [samples, classes]."
        )

    validate_finite_array(
        logits,
        name="logits",
    )

    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(logits, -500, 500))
    )

    max_probability = np.max(
        probabilities,
        axis=1,
    )

    scores = 1.0 - max_probability

    return validate_ood_scores(scores)


def energy_score(
    logits,
    temperature=ENERGY_TEMPERATURE,
):
    """
    Calculate the standard energy-based OOD score.

    Formula:
        E(x) = -T * logsumexp(logits / T)

    Higher scores (less negative energy) indicate
    more OOD-like inputs.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    if logits.ndim != 2:
        raise ValueError(
            "Logits must have shape [samples, classes]."
        )

    validate_finite_array(
        logits,
        name="logits",
    )

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "Temperature must be finite and greater than zero."
        )

    scaled = logits / temperature

    max_logits = np.max(
        scaled,
        axis=1,
        keepdims=True,
    )

    stable_logsumexp = (
        max_logits
        + np.log(
            np.sum(
                np.exp(scaled - max_logits),
                axis=1,
                keepdims=True,
            )
        )
    )

    scores = (
        -temperature * stable_logsumexp
    ).reshape(-1)

    return validate_ood_scores(scores)

def mahalanobis_score(
    features,
    mean,
    safe_variance,
):
    """
    Calculate diagonal Mahalanobis distance.

    Formula:

        D(x) =
            sum(
                (x_j - mean_j)^2
                /
                variance_j
            )

    Higher distance = more OOD-like.
    """

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    mean = np.asarray(
        mean,
        dtype=np.float64,
    )

    safe_variance = np.asarray(
        safe_variance,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "Features must have shape [samples, features]."
        )

    if mean.ndim != 1:
        raise ValueError(
            "Mean must be a 1-dimensional vector."
        )

    if safe_variance.ndim != 1:
        raise ValueError(
            "Safe variance must be a 1-dimensional vector."
        )

    if features.shape[1] != mean.shape[0]:
        raise ValueError(
            "Feature dimension and mean dimension do not match."
        )

    if features.shape[1] != safe_variance.shape[0]:
        raise ValueError(
            "Feature dimension and variance dimension do not match."
        )

    validate_finite_array(
        features,
        name="features",
    )

    validate_finite_array(
        mean,
        name="mean",
    )

    validate_finite_array(
        safe_variance,
        name="safe_variance",
    )

    if np.any(safe_variance <= 0):
        raise ValueError(
            "All safe variance values must be positive."
        )

    centered = features - mean.reshape(1, -1)

    squared_distance = centered ** 2

    normalized_distance = (
        squared_distance
        / safe_variance.reshape(1, -1)
    )

    scores = np.sum(
        normalized_distance,
        axis=1,
    )

    return validate_ood_scores(scores)


def fit_mahalanobis_statistics(
    features,
    variance_epsilon=1e-6,
):
    """
    Fit diagonal Mahalanobis statistics from in-distribution
    reference features.

    Parameters:
        features:
            NumPy array with shape [samples, features].
            These must be extracted from representative,
            in-distribution reference images.

        variance_epsilon:
            Minimum variance to prevent division by zero
            and unstable distance calculations.

    Returns:
        mean:
            Feature-wise mean vector.

        safe_variance:
            Feature-wise variance vector with a minimum
            variance floor.

    Notes:
        This implementation estimates a diagonal covariance
        approximation, not a full covariance matrix.
    """

    features = np.asarray(
        features,
        dtype=np.float64,
    )

    if features.ndim != 2:
        raise ValueError(
            "Features must have shape [samples, features]."
        )

    if features.shape[0] < 2:
        raise ValueError(
            "At least two reference samples are required."
        )

    if features.shape[1] < 1:
        raise ValueError(
            "Features must contain at least one dimension."
        )

    validate_finite_array(
        features,
        name="features",
    )

    if (
        not np.isfinite(variance_epsilon)
        or variance_epsilon <= 0
    ):
        raise ValueError(
            "variance_epsilon must be finite and positive."
        )

    mean = np.mean(
        features,
        axis=0,
    )

    variance = np.var(
        features,
        axis=0,
        ddof=1,
    )

    safe_variance = np.maximum(
        variance,
        variance_epsilon,
    )

    return mean, safe_variance



def sigmoid(logits):
    """
    Convert logits to sigmoid probabilities.
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    validate_finite_array(
        logits,
        name="logits",
    )

    return 1.0 / (
        1.0 + np.exp(-np.clip(logits, -500, 500))
    )


def score_all_methods(
    logits,
    features=None,
    mean=None,
    safe_variance=None,
):
    """
    Calculate all available OOD scores.

    Mahalanobis is calculated only when feature statistics
    are supplied.
    """

    results = {}

    results["msp"] = msp_score(
        logits
    )

    results["energy"] = energy_score(
        logits
    )

    if (
        features is not None
        and mean is not None
        and safe_variance is not None
    ):
        results["mahalanobis"] = mahalanobis_score(
            features=features,
            mean=mean,
            safe_variance=safe_variance,
        )

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("MEDTRUST OOD SCORING")
    print("=" * 70)

    example_logits = np.array(
        [
            [4.0, -2.0, -1.0],
            [0.1, 0.2, 0.0],
        ]
    )

    print("MSP scores:")
    print(msp_score(example_logits))

    print()
    print("Energy scores:")
    print(energy_score(example_logits))

    print()
    print("Scoring test completed.")