"""
MedTrust V1 — Module 5: Dataset-Based Quality Reference Fitting
(src/quality/fit_quality_reference.py)

Calculates reference ranges for image-quality metrics from chest radiograph datasets
using empirical percentiles (5th and 95th percentiles).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from .image_quality_analyzer import calculate_raw_metrics, load_grayscale_image
from .quality_reference import (
    DEFAULT_REFERENCE_PATH,
    PROJECT_ROOT,
    QualityReference,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LOWER_PERCENTILE = 5.0
UPPER_PERCENTILE = 95.0


def find_images(dataset_path: Path) -> List[Path]:
    """Find all supported image files recursively."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if dataset_path.is_file():
        return [dataset_path]

    images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(dataset_path.rglob(f"*{ext}"))
        images.extend(dataset_path.rglob(f"*{ext.upper()}"))
    return sorted(list(set(images)))


def collect_metrics(image_paths: List[Path]) -> Dict[str, List[float]]:
    """Compute raw quality metrics across all provided images."""
    data: Dict[str, List[float]] = {
        "sharpness": [],
        "contrast": [],
        "brightness": [],
        "clipping": [],
        "noise": [],
        "resolution": [],
    }

    total = len(image_paths)
    for idx, path in enumerate(image_paths, 1):
        try:
            gray = load_grayscale_image(path)
            raw = calculate_raw_metrics(gray)
            for k in data:
                data[k].append(raw[k])
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

        if idx % 100 == 0 or idx == total:
            print(f"Processed {idx}/{total} images...", end="\r")
    print()
    return data


def compute_statistics(metrics_data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Calculate summary percentiles and distributions for each metric."""
    stats = {}
    for name, values in metrics_data.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"No valid metric values collected for {name}.")
        stats[name] = {
            "count": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, LOWER_PERCENTILE)),
            "p95": float(np.percentile(arr, UPPER_PERCENTILE)),
        }
    return stats


def fit_quality_reference(stats: Dict[str, Dict[str, float]]) -> QualityReference:
    """Create QualityReference from computed percentiles."""
    b_med = stats["brightness"]["median"]
    b_p05 = stats["brightness"]["p05"]
    b_p95 = stats["brightness"]["p95"]
    b_tol = max(abs(b_med - b_p05), abs(b_p95 - b_med), 40.0)

    return QualityReference(
        sharpness_low=max(float(stats["sharpness"]["p05"]), 1.0),
        sharpness_high=max(float(stats["sharpness"]["p95"]), 30.0),
        contrast_low=max(float(stats["contrast"]["p05"]), 5.0),
        contrast_high=max(float(stats["contrast"]["p95"]), 40.0),
        brightness_target=float(b_med),
        brightness_tolerance=float(b_tol),
        clipping_low=0.0,
        clipping_high=max(float(stats["clipping"]["p95"]), 15.0),
        noise_low=0.0,
        noise_high=max(float(stats["noise"]["p95"]), 5.0),
        resolution_low=min(float(stats["resolution"]["p05"]), 250.0),
        resolution_high=max(float(stats["resolution"]["p95"]), 500.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit chest X-ray quality reference from image dataset.")
    parser.add_argument("dataset_dir", type=str, help="Directory containing sample images.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_REFERENCE_PATH), help="Output reference JSON path.")
    parser.add_argument("--stats-output", type=str, default=None, help="Output statistics JSON path.")
    args = parser.parse_args()

    images = find_images(Path(args.dataset_dir))
    if not images:
        print(f"No image files found in {args.dataset_dir}")
        sys.exit(1)

    print(f"Fitting reference across {len(images)} images...")
    metrics_data = collect_metrics(images)
    stats = compute_statistics(metrics_data)
    ref = fit_quality_reference(stats)

    out_p = Path(args.output)
    ref.save(out_p)
    print(f"Saved fitted reference to: {out_p}")

    stats_p = Path(args.stats_output) if args.stats_output else out_p.parent / "quality_statistics.json"
    with open(stats_p, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved statistics to: {stats_p}")


if __name__ == "__main__":
    main()

