"""
MedTrust Out-of-Distribution (OOD) Detection Module (src/OOD/__init__.py)

This module provides post-hoc OOD detection for the frozen MedTrust DenseNet-121 classifier.

Supported methods:
    - Energy score (Liu et al., 2020)
    - MSP score (Hendrycks & Gimpel, 2017)
    - Mahalanobis feature distance (Lee et al., 2018)

Produces conservative reliability status:
    - in-distribution-like
    - borderline/uncertain
    - out-of-distribution-like
per MedTrust AGENTS.md Rule 17.
"""

__version__ = "1.0.0"

from .config import (
    CHECKPOINT_PATH,
    OOD_DATA_DIR,
    OOD_OUTPUT_DIR,
    NUM_CLASSES,
    FEATURE_DIM,
)
from .detector import (
    OODDetector,
    OODResult,
)
from .scores import (
    energy_score,
    msp_score,
    mahalanobis_score,
    fit_mahalanobis_statistics,
    score_all_methods,
)

__all__ = [
    "OODDetector",
    "OODResult",
    "energy_score",
    "msp_score",
    "mahalanobis_score",
    "fit_mahalanobis_statistics",
    "score_all_methods",
    "CHECKPOINT_PATH",
    "OOD_DATA_DIR",
    "OOD_OUTPUT_DIR",
    "NUM_CLASSES",
    "FEATURE_DIM",
]