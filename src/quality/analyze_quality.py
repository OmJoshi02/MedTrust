"""
MedTrust V1 — Module 5: Image Quality Analysis CLI (src/quality/analyze_quality.py)

Usage:
    python -m src.quality.analyze_quality test_images/disease.png --detailed
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .image_quality_analyzer import (
    WeightedImageQualityAnalyzer,
    print_quality_report,
)
from .quality_reference import DEFAULT_REFERENCE_PATH, QualityReference


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedTrust Module 5: Technical Radiograph Quality Analysis"
    )
    parser.add_argument("image_path", type=str, help="Path to chest X-ray image.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional path to calibrated quality reference JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Acceptance threshold (0-100, default: 60.0).",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Display detailed breakdown of metrics and warnings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    ref = None
    if args.reference:
        ref = QualityReference.load(args.reference)

    analyzer = WeightedImageQualityAnalyzer(
        reference=ref,
        acceptance_threshold=args.threshold,
    )

    try:
        result = analyzer.analyze(args.image_path)
        print_quality_report(result, detailed=args.detailed)
    except Exception as exc:
        print(f"Error during image quality analysis: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

