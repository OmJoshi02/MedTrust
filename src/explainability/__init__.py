"""
MedTrust V1 — Explainability & Reliability Package (src/explainability)
"""

from pathlib import Path

from src.explainability.gradcam import GradCAM
from src.explainability.reliability import ExplanationReliabilityAnalyzer

DEFAULT_GRADCAM_OUTPUT_DIR = Path("outputs/explainability/gradcam")
GRADCAM_OUTPUT_DIR = DEFAULT_GRADCAM_OUTPUT_DIR

__all__ = [
    "GradCAM",
    "ExplanationReliabilityAnalyzer",
    "DEFAULT_GRADCAM_OUTPUT_DIR",
    "GRADCAM_OUTPUT_DIR",
]
