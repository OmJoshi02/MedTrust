"""
MedTrust V1 — Module 5: Quality Reference Configuration (src/quality/quality_reference.py)

Defines reference ranges used for normalizing chest X-ray image-quality metrics.
Reference values are engineering benchmarks derived from chest radiograph distributions.
They are NOT clinical certification standards (AGENTS.md Rule 15).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

# Anchor to MedTrust project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_PATH = PROJECT_ROOT / "outputs" / "quality" / "quality_reference.json"


@dataclass
class QualityReference:
    """
    Reference configuration for technical image-quality normalization.
    Converts raw technical metrics into normalized [0, 1] scores.
    """

    # 1. Sharpness (Variance of Laplacian)
    sharpness_low: float = 10.0
    sharpness_high: float = 160.0

    # 2. Contrast (Grayscale standard deviation)
    contrast_low: float = 20.0
    contrast_high: float = 80.0

    # 3. Brightness (Mean grayscale intensity: target centered)
    brightness_target: float = 160.0
    brightness_tolerance: float = 70.0

    # 4. Clipping (% pixels <= 5 or >= 250)
    clipping_low: float = 0.0
    clipping_high: float = 30.0

    # 5. Noise (Median-filter residual standard deviation)
    noise_low: float = 0.0
    noise_high: float = 15.0

    # 6. Resolution (Minimum spatial dimension in pixels)
    resolution_low: float = 250.0
    resolution_high: float = 1200.0

    def __post_init__(self) -> None:
        """Validate the reference range configuration."""
        self._validate_range(self.sharpness_low, self.sharpness_high, "sharpness")
        self._validate_range(self.contrast_low, self.contrast_high, "contrast")
        self._validate_range(self.clipping_low, self.clipping_high, "clipping")
        self._validate_range(self.noise_low, self.noise_high, "noise")
        self._validate_range(self.resolution_low, self.resolution_high, "resolution")

        if not (0.0 <= self.brightness_target <= 255.0):
            raise ValueError("brightness_target must be between 0.0 and 255.0.")

        if self.brightness_tolerance <= 0.0:
            raise ValueError("brightness_tolerance must be strictly positive.")

    @staticmethod
    def _validate_range(low: float, high: float, name: str) -> None:
        """Ensure reference low/high bounds are well-formed."""
        if low < 0.0:
            raise ValueError(f"{name}_low cannot be negative (got {low}).")
        if high <= low:
            raise ValueError(f"{name}_high must be strictly greater than {name}_low (got low={low}, high={high}).")

    def to_dict(self) -> dict:
        """Convert reference configuration into a standard dictionary."""
        return asdict(self)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save reference configuration as JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "QualityReference":
        """
        Load reference configuration from JSON.
        Falls back to DEFAULT_REFERENCE_PATH if filepath is omitted.
        """
        p = Path(filepath) if filepath is not None else DEFAULT_REFERENCE_PATH
        if not p.exists():
            raise FileNotFoundError(f"Quality reference file not found at: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(**data)


def create_default_reference() -> QualityReference:
    """Create default engineering reference configuration."""
    return QualityReference()

