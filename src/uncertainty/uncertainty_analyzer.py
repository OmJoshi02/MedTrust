"""
High-level uncertainty analysis for MC-Dropout predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .uncertainty_metrics import uncertainty_summary


@dataclass
class DiseaseUncertainty:
    disease: str
    probability: float
    variance: float
    standard_deviation: float
    entropy: float
    variation_ratio: float
    confidence: float


class UncertaintyAnalyzer:
    """
    Convert MC-Dropout samples into human-readable disease-level results.

    This class does not fit thresholds and does not modify the model.
    """

    def __init__(self, classes: Sequence[str]) -> None:
        self.classes = list(classes)

        if not self.classes:
            raise ValueError("classes must not be empty.")

    def analyze(
        self,
        probability_samples: np.ndarray,
        decision_threshold: float = 0.5,
    ) -> dict:
        probability_samples = np.asarray(
            probability_samples,
            dtype=np.float64,
        )

        if probability_samples.ndim != 2:
            raise ValueError(
                "probability_samples must have shape [passes, diseases]."
            )

        if probability_samples.shape[1] != len(self.classes):
            raise ValueError(
                "Number of diseases in samples does not match classes."
            )

        if not np.isfinite(probability_samples).all():
            raise ValueError("probability_samples contains non-finite values.")

        if np.any((probability_samples < 0.0) | (probability_samples > 1.0)):
            raise ValueError(
                "probability_samples must be within [0, 1]."
            )

        metrics = uncertainty_summary(
            probability_samples,
            decision_threshold=decision_threshold,
        )

        diseases = []

        for index, disease in enumerate(self.classes):
            diseases.append(
                DiseaseUncertainty(
                    disease=disease,
                    probability=float(metrics["mean_probability"][index]),
                    variance=float(metrics["predictive_variance"][index]),
                    standard_deviation=float(metrics["predictive_std"][index]),
                    entropy=float(metrics["normalized_entropy"][index]),
                    variation_ratio=float(metrics["variation_ratio"][index]),
                    confidence=float(metrics["confidence"][index]),
                )
            )

        most_uncertain_index = int(
            np.argmax(metrics["predictive_variance"])
        )

        return {
            "passes": int(probability_samples.shape[0]),
            "diseases": diseases,
            "overall_mean_variance": metrics["overall_mean_variance"],
            "overall_max_variance": metrics["overall_max_variance"],
            "overall_mean_entropy": metrics["overall_mean_entropy"],
            "overall_max_entropy": metrics["overall_max_entropy"],
            "most_uncertain_disease": self.classes[most_uncertain_index],
            "most_uncertain_variance": float(
                metrics["predictive_variance"][most_uncertain_index]
            ),
        }

    @staticmethod
    def trust_label(
        overall_mean_variance: float,
        low_threshold: float,
        high_threshold: float,
    ) -> str:
        """
        Convert an uncertainty score to a descriptive status.

        The thresholds are NOT medical standards. They should be selected
        using the validation set and frozen before test evaluation.
        """
        if low_threshold >= high_threshold:
            raise ValueError(
                "low_threshold must be smaller than high_threshold."
            )

        if overall_mean_variance <= low_threshold:
            return "LOW_UNCERTAINTY"

        if overall_mean_variance <= high_threshold:
            return "MODERATE_UNCERTAINTY"

        return "HIGH_UNCERTAINTY"
