"""
Mathematical metrics used by the MedTrust Uncertainty Analyzer.
"""

from __future__ import annotations

import numpy as np


def predictive_variance(samples: np.ndarray) -> np.ndarray:
    """
    Predictive variance across MC samples.

    Formula:
        Var[p] = (1/T) * sum_t (p_t - mean(p))^2

    Parameters
    ----------
    samples:
        Array shaped [passes, diseases].
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError("samples must have shape [passes, diseases].")

    return np.var(samples, axis=0, ddof=0)


def predictive_std(samples: np.ndarray) -> np.ndarray:
    """Square root of predictive variance."""
    return np.sqrt(predictive_variance(samples))


def binary_entropy(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Binary predictive entropy for independent multi-label outputs.

    Formula:
        H(p) = -p log(p) - (1-p) log(1-p)

    The natural-log version is used. Maximum entropy is ln(2).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = np.clip(probabilities, eps, 1.0 - eps)

    return -(
        probabilities * np.log(probabilities)
        + (1.0 - probabilities) * np.log(1.0 - probabilities)
    )


def normalized_binary_entropy(
    probabilities: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Binary entropy normalized to approximately [0, 1]."""
    return binary_entropy(probabilities, eps=eps) / np.log(2.0)


def variation_ratio(
    samples: np.ndarray,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """
    Variation ratio of binary MC decisions.

    Formula:
        VR = 1 - max(count(class)) / T

    With binary outputs, this is the fraction of stochastic passes that
    disagree with the majority decision.
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError("samples must have shape [passes, diseases].")

    decisions = samples >= decision_threshold
    positive_fraction = decisions.mean(axis=0)

    return 1.0 - np.maximum(
        positive_fraction,
        1.0 - positive_fraction,
    )


def mean_confidence(probabilities: np.ndarray) -> np.ndarray:
    """
    Confidence proxy for multi-label predictions.

    For each disease, confidence is the distance of the mean probability
    from the maximally ambiguous value 0.5:
        confidence = 2 * |p - 0.5|

    This is deliberately kept separate from uncertainty.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return 2.0 * np.abs(probabilities - 0.5)


def uncertainty_summary(
    samples: np.ndarray,
    decision_threshold: float = 0.5,
) -> dict:
    """Calculate per-disease and image-level uncertainty statistics."""
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim != 2:
        raise ValueError("samples must have shape [passes, diseases].")

    mean_probability = samples.mean(axis=0)
    variance = predictive_variance(samples)
    std = np.sqrt(variance)
    entropy = normalized_binary_entropy(mean_probability)
    vr = variation_ratio(
        samples,
        decision_threshold=decision_threshold,
    )

    return {
        "mean_probability": mean_probability,
        "predictive_variance": variance,
        "predictive_std": std,
        "normalized_entropy": entropy,
        "variation_ratio": vr,
        "confidence": mean_confidence(mean_probability),
        "overall_mean_variance": float(np.mean(variance)),
        "overall_max_variance": float(np.max(variance)),
        "overall_mean_entropy": float(np.mean(entropy)),
        "overall_max_entropy": float(np.max(entropy)),
    }
