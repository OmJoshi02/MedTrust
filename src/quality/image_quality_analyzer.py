"""
MedTrust V1 — Module 5: Weighted Scoring-Based Image Quality Analyzer
(src/quality/image_quality_analyzer.py)

Evaluates chest X-ray technical radiograph quality using six fundamental indicators:
    1. Sharpness  (Weight: 0.25) — Edge definition via Laplacian variance
    2. Contrast   (Weight: 0.20) — Intensity spread via standard deviation
    3. Brightness (Weight: 0.15) — Target-centered mean luminance (penetration proxy)
    4. Clipping   (Weight: 0.15) — Sensor saturation at extreme black/white levels
    5. Noise      (Weight: 0.15) — Quantum mottle estimate via median filter residual
    6. Resolution (Weight: 0.10) — Spatial dimension fidelity min(height, width)

Scientific Notice:
Quality outputs are engineering reliability indicators for decision support.
They do NOT constitute clinical image-quality certification (AGENTS.md Rule 15 & 28).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .quality_reference import (
    DEFAULT_REFERENCE_PATH,
    QualityReference,
    create_default_reference,
)

# Canonical metric weights (sum to 1.00)
WEIGHTS: Dict[str, float] = {
    "sharpness": 0.25,
    "contrast": 0.20,
    "brightness": 0.15,
    "clipping": 0.15,
    "noise": 0.15,
    "resolution": 0.10,
}


@dataclass
class QualityResult:
    """Stores the complete technical image-quality analysis result."""

    quality_score: float
    status: str
    raw_metrics: Dict[str, float]
    normalized_metrics: Dict[str, float]
    weighted_contributions: Dict[str, float]
    warnings: List[str]
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a JSON-serializable dictionary."""
        return {
            "quality_score": round(self.quality_score, 2),
            "composite_score": int(round(self.quality_score)),
            "status": self.status,
            "quality_status": "Good Quality" if self.quality_score >= 75.0 else ("Acceptable Quality" if self.quality_score >= 55.0 else "Low Quality"),
            "suitable": self.status == "ACCEPTED",
            "raw_metrics": {k: round(v, 4) for k, v in self.raw_metrics.items()},
            "normalized_metrics": {k: round(v, 4) for k, v in self.normalized_metrics.items()},
            "weighted_contributions": {k: round(v, 4) for k, v in self.weighted_contributions.items()},
            "sharpness": round(self.normalized_metrics["sharpness"] * 100.0, 1),
            "contrast": round(self.normalized_metrics["contrast"] * 100.0, 1),
            "brightness": round(self.normalized_metrics["brightness"] * 100.0, 1),
            "clipping": round(self.raw_metrics["clipping"], 2),
            "noise": round(self.raw_metrics["noise"], 2),
            "resolution": f"{self.width} x {self.height}",
            "warnings": list(self.warnings),
            "width": self.width,
            "height": self.height,
        }


# ============================================================
# IMAGE LOADING & NORMALIZATION
# ============================================================

def load_grayscale_image(
    image_input: Union[str, Path, Image.Image, np.ndarray],
) -> np.ndarray:
    """
    Polymorphic image loader: loads or converts any input to a 2D uint8 grayscale numpy array.
    Supports file paths (str, Path), PIL Images, and NumPy arrays (grayscale or RGB).
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image file not found: {p}")
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to PIL in case cv2 has format issues
            try:
                with Image.open(p) as pil_img:
                    return np.array(pil_img.convert("L"), dtype=np.uint8)
            except Exception as e:
                raise ValueError(f"Failed to load image from {p}: {e}")
        return img

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("L"), dtype=np.uint8)

    if isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.size == 0:
            raise ValueError("Input image array is empty.")

        # Convert float [0, 1] to uint8 [0, 255]
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 + 1e-6:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Handle 3D (H, W, C) images
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            elif arr.shape[2] == 1:
                return arr.squeeze(axis=2).astype(np.uint8)

        if arr.ndim == 2:
            return arr.astype(np.uint8)

        raise ValueError(f"Unexpected image array shape: {arr.shape}")

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


# ============================================================
# TECHNICAL METRIC CALCULATIONS
# ============================================================

def calculate_sharpness(gray_img: np.ndarray) -> float:
    """Calculate image sharpness via normalized Variance of Laplacian: Var(Laplacian(I))."""
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return float(laplacian.var())


def calculate_contrast(gray_img: np.ndarray) -> float:
    """Calculate image contrast via grayscale intensity standard deviation: std(I)."""
    return float(np.std(gray_img))


def calculate_brightness(gray_img: np.ndarray) -> float:
    """Calculate mean grayscale intensity: mean(I) in [0, 255]."""
    return float(np.mean(gray_img))


def calculate_clipping(
    gray_img: np.ndarray,
    dark_threshold: int = 5,
    bright_threshold: int = 250,
) -> float:
    """
    Calculate the percentage of pixels clipped at extreme black or white levels:
    Clipping % = (count(I <= 5) + count(I >= 250)) / total_pixels * 100.
    """
    dark_pixels = np.sum(gray_img <= dark_threshold)
    bright_pixels = np.sum(gray_img >= bright_threshold)
    total_pixels = gray_img.size
    return float((dark_pixels + bright_pixels) / total_pixels * 100.0)


def calculate_noise(gray_img: np.ndarray) -> float:
    """
    Estimate image noise level via high-frequency residual from a 3x3 median filter:
    Residual = I - MedianFilter(I); Noise ≈ std(Residual).
    """
    denoised = cv2.medianBlur(gray_img, 3)
    residual = gray_img.astype(np.float32) - denoised.astype(np.float32)
    return float(np.std(residual))


def calculate_resolution(gray_img: np.ndarray) -> float:
    """Calculate minimum spatial dimension: min(height, width)."""
    h, w = gray_img.shape[:2]
    return float(min(h, w))


def calculate_raw_metrics(gray_img: np.ndarray) -> Dict[str, float]:
    """Calculate all six raw technical quality metrics."""
    return {
        "sharpness": calculate_sharpness(gray_img),
        "contrast": calculate_contrast(gray_img),
        "brightness": calculate_brightness(gray_img),
        "clipping": calculate_clipping(gray_img),
        "noise": calculate_noise(gray_img),
        "resolution": calculate_resolution(gray_img),
    }


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_higher_is_better(value: float, low: float, high: float) -> float:
    """Normalize a metric where higher values represent superior technical quality."""
    if high <= low:
        raise ValueError(f"High reference bound ({high}) must be strictly greater than low ({low}).")
    normalized = (value - low) / (high - low)
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_lower_is_better(value: float, low: float, high: float) -> float:
    """Normalize a metric where lower values represent superior technical quality (e.g. clipping, noise)."""
    if high <= low:
        raise ValueError(f"High reference bound ({high}) must be strictly greater than low ({low}).")
    normalized = 1.0 - (value - low) / (high - low)
    return float(np.clip(normalized, 0.0, 1.0))


def normalize_brightness(value: float, target: float, tolerance: float) -> float:
    """
    Normalize brightness using target-centered penetration reference:
    Score = 1.0 - |value - target| / tolerance, clipped to [0, 1].
    """
    if tolerance <= 0:
        raise ValueError("Brightness tolerance must be strictly greater than zero.")
    distance = abs(value - target)
    score = 1.0 - (distance / tolerance)
    return float(np.clip(score, 0.0, 1.0))


def normalize_metrics(
    raw_metrics: Dict[str, float],
    reference: QualityReference,
) -> Dict[str, float]:
    """Convert all raw technical metrics into normalized [0, 1] scores using reference ranges."""
    return {
        "sharpness": normalize_higher_is_better(
            raw_metrics["sharpness"], reference.sharpness_low, reference.sharpness_high
        ),
        "contrast": normalize_higher_is_better(
            raw_metrics["contrast"], reference.contrast_low, reference.contrast_high
        ),
        "brightness": normalize_brightness(
            raw_metrics["brightness"], reference.brightness_target, reference.brightness_tolerance
        ),
        "clipping": normalize_lower_is_better(
            raw_metrics["clipping"], reference.clipping_low, reference.clipping_high
        ),
        "noise": normalize_lower_is_better(
            raw_metrics["noise"], reference.noise_low, reference.noise_high
        ),
        "resolution": normalize_higher_is_better(
            raw_metrics["resolution"], reference.resolution_low, reference.resolution_high
        ),
    }


# ============================================================
# COMPOSITE SCORING & WARNING GENERATION
# ============================================================

def calculate_weighted_score(
    normalized_metrics: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted composite quality score Q in [0, 100].
    Returns final score and individual weighted contributions.
    """
    contributions = {
        metric: normalized_metrics[metric] * weight
        for metric, weight in WEIGHTS.items()
    }
    normalized_score = sum(contributions.values())
    quality_score = float(normalized_score * 100.0)
    return quality_score, contributions


def determine_status(quality_score: float, threshold: float = 55.0) -> str:
    """Determine acceptance status based on predefined technical threshold."""
    return "ACCEPTED" if quality_score >= threshold else "REJECTED"


def generate_warnings(
    raw_metrics: Dict[str, float],
    normalized_metrics: Dict[str, float],
) -> List[str]:
    """
    Generate human-interpretable technical quality warnings.
    Includes the essential clinical pathology safeguard (AGENTS.md Rule 15).
    """
    warnings: List[str] = []

    if normalized_metrics["sharpness"] < 0.30:
        warnings.append("Low sharpness detected. Radiograph may contain reduced edge definition.")

    if normalized_metrics["contrast"] < 0.30:
        warnings.append("Low contrast detected. Dynamic intensity differences between tissues may be limited.")

    if normalized_metrics["brightness"] < 0.30:
        warnings.append("Brightness / exposure deviates substantially from the calibrated reference range.")

    if normalized_metrics["clipping"] < 0.30:
        warnings.append("High pixel clipping detected. Anatomical data may be saturated in dense white or pure black regions.")

    if normalized_metrics["noise"] < 0.30:
        warnings.append("Elevated high-frequency noise / quantum mottle detected.")

    if normalized_metrics["resolution"] < 0.30:
        warnings.append("Low image resolution detected (< 250px spatial dimension).")

    # Clinical pathology safeguard: dense consolidation/effusion can reduce edge variance without being technical blur
    if (
        normalized_metrics["sharpness"] < 0.30
        and normalized_metrics["contrast"] >= 0.50
        and normalized_metrics["brightness"] >= 0.50
        and normalized_metrics["noise"] >= 0.50
    ):
        warnings.append(
            "Low sharpness may be influenced by anatomical content or pathology (e.g. extensive consolidation); "
            "technical blur cannot be concluded from sharpness alone."
        )

    return warnings


# ============================================================
# MAIN QUALITY ANALYZER CLASS
# ============================================================

class WeightedImageQualityAnalyzer:
    """
    Unified Image Quality Analyzer for MedTrust (Module 5).
    Evaluates, normalizes, and aggregates radiograph technical quality indicators.
    """

    def __init__(
        self,
        reference: Optional[QualityReference] = None,
        acceptance_threshold: float = 55.0,
    ) -> None:
        if reference is None:
            try:
                reference = QualityReference.load(DEFAULT_REFERENCE_PATH)
            except Exception:
                reference = create_default_reference()

        self.reference = reference
        self.acceptance_threshold = float(acceptance_threshold)

    def analyze(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> QualityResult:
        """
        Perform complete technical image quality assessment on any supported input.
        """
        gray_img = load_grayscale_image(image_input)
        height, width = gray_img.shape[:2]

        raw_metrics = calculate_raw_metrics(gray_img)
        normalized_metrics = normalize_metrics(raw_metrics, self.reference)
        quality_score, weighted_contributions = calculate_weighted_score(normalized_metrics)
        status = determine_status(quality_score, threshold=self.acceptance_threshold)
        warnings = generate_warnings(raw_metrics, normalized_metrics)

        return QualityResult(
            quality_score=quality_score,
            status=status,
            raw_metrics=raw_metrics,
            normalized_metrics=normalized_metrics,
            weighted_contributions=weighted_contributions,
            warnings=warnings,
            width=width,
            height=height,
        )


def print_quality_report(result: QualityResult, detailed: bool = False) -> None:
    """Print readable ASCII report of quality evaluation."""
    print("=" * 72)
    print("MEDTRUST — MODULE 5: IMAGE QUALITY REPORT")
    print("=" * 72)
    print(f"Status           : {result.status}")
    print(f"Quality Score    : {result.quality_score:.2f} / 100.0")
    print(f"Resolution       : {result.width} x {result.height}")

    if detailed:
        print("\nRAW TECHNICAL METRICS")
        print("-" * 72)
        for k, v in result.raw_metrics.items():
            print(f"  {k.capitalize():15s}: {v:.4f}")

        print("\nNORMALIZED METRICS [0 - 1]")
        print("-" * 72)
        for k, v in result.normalized_metrics.items():
            print(f"  {k.capitalize():15s}: {v:.4f}")

        print("\nWEIGHTED CONTRIBUTIONS")
        print("-" * 72)
        for k, v in result.weighted_contributions.items():
            print(f"  {k.capitalize():15s}: {v:.4f}")

    if result.warnings:
        print("\nWARNINGS / CAVEATS")
        print("-" * 72)
        for w in result.warnings:
            print(f"  - {w}")
    else:
        print("\nWarnings         : None (Satisfies nominal technical quality limits)")
    print("=" * 72)
