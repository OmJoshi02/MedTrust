"""
MedTrust V1 — Dedicated Script to Generate NIH ChestX-ray14 Disease Frequency Artifacts
(scripts/generate_disease_frequency.py)

Module 8: Similarity Analysis.
Parses dataset/Data_Entry_2017_v2020.csv, computes exact disease frequencies and prevalences,
and writes structured JSON and CSV artifacts to outputs/similarity/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity.disease_frequency import (
    DEFAULT_NIH_CSV,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_CSV,
    compute_disease_frequencies,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate NIH ChestX-ray14 Disease Frequency and Prevalence Artifacts"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_NIH_CSV),
        help=f"Path to NIH Data_Entry CSV (default: {DEFAULT_NIH_CSV})",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(DEFAULT_OUTPUT_JSON),
        help=f"Path for output JSON artifact (default: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help=f"Path for output CSV artifact (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[Error] Dataset CSV not found at: {csv_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("=" * 70)
        print("  MEDTRUST V1 — GENERATING NIH DISEASE PREVALENCE ARTIFACTS")
        print("=" * 70)
        print(f"Reading Dataset from : {csv_path}")

    res = compute_disease_frequencies(
        csv_path=csv_path,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
    )

    if not args.quiet:
        print(f"Total Valid Records  : {res['total_valid_records']:,}")
        print(f"Duplicates Excluded  : {res['duplicate_records_count']}")
        print(f"JSON Artifact Saved  : {res['output_json_path']}")
        print(f"CSV Artifact Saved   : {res['output_csv_path']}")
        print("-" * 70)
        print(f"{'Disease Class':<24} {'Count':>10} {'Prevalence':>14}")
        print("-" * 70)
        for rec in res["diseases"]:
            print(f"{rec['disease_name']:<24} {rec['occurrence_count']:>10,} {rec['prevalence_percentage']:>13.4f}%")
        print("-" * 70)
        nf = res["no_finding"]
        print(f"{nf['disease_name']:<24} {nf['occurrence_count']:>10,} {nf['prevalence_percentage']:>13.4f}%")
        print("=" * 70)
        print("[Success] NIH disease prevalence artifacts successfully generated.")


if __name__ == "__main__":
    main()

