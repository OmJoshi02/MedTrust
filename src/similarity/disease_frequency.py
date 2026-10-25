"""
MedTrust V1 — NIH ChestX-ray14 Disease Frequency & Prevalence Analysis
(src/similarity/disease_frequency.py)

Module 8: Similarity Analysis.
Calculates label frequencies and global dataset prevalence from the official
NIH ChestX-ray14 master catalog (Data_Entry_2017_v2020.csv).

Mathematical Formulation:
    For disease d, let Y_i be the set of labels for valid unique image record i (1 <= i <= N):
        Frequency:   F_d = sum_{i=1}^N 1(d in Y_i)
        Prevalence:  P_d = (F_d / N) * 100.0 (%)

Methodological Invariants:
1. Each disease is counted at most once per image record.
2. Multilabel records contribute to each labeled disease independently (non-mutually exclusive).
3. "No Finding" is reported separately and excluded from the 14 disease-class counts.
4. Denominator N represents unique, valid image records. Duplicates and malformed rows are tracked.
5. Global disease prevalence is dataset context only; it is NOT a query prediction probability.
6. Strictly fails with FileNotFoundError if the CSV catalog is unavailable (zero fabricated metrics).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_NIH_CSV = PROJECT_ROOT / "dataset" / "Data_Entry_2017_v2020.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "similarity" / "disease_frequency.csv"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "outputs" / "similarity" / "disease_frequency.json"

# Canonical 14 disease classes in fixed MedTrust V1 order (AGENTS.md Rule 3)
CANONICAL_DISEASES: Tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
)

# Mandatory CSV columns that must exist in NIH catalog
MANDATORY_CSV_COLUMNS: Tuple[str, ...] = (
    "Image Index",
    "Finding Labels",
    "Patient Age",
    "Patient Gender",
    "View Position",
)

NO_FINDING_LABEL = "No Finding"

# In-memory lifecycle caches
_CACHED_JSON_DATA: Optional[Dict[str, Any]] = None
_CACHED_CSV_TABLE: Optional[Dict[str, Dict[str, Any]]] = None


def parse_labels_string(raw_labels: Optional[str]) -> Set[str]:
    """
    Parse pipe-separated or comma-separated finding labels into a set of normalized strings.
    Handles 'No Finding', empty strings, and delimiters consistently.
    """
    if raw_labels is None or str(raw_labels).strip() in ("", "None", "nan", "NaN"):
        return {NO_FINDING_LABEL}

    text = str(raw_labels).strip()
    delimiters = ["|", ","]
    for d in delimiters:
        if d in text:
            parts = {p.strip() for p in text.split(d) if p.strip()}
            return parts if parts else {NO_FINDING_LABEL}

    return {text}


def compute_disease_frequencies(
    csv_path: Union[str, Path] = DEFAULT_NIH_CSV,
    output_json_path: Optional[Union[str, Path]] = DEFAULT_OUTPUT_JSON,
    output_csv_path: Optional[Union[str, Path]] = DEFAULT_OUTPUT_CSV,
    dataset_source_name: str = "NIH ChestX-ray14 (Data_Entry_2017_v2020.csv)",
) -> Dict[str, Any]:
    """
    Parse the NIH master CSV and compute exact disease frequencies and prevalences.

    Validates:
    - Existence of mandatory columns ('Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender', 'View Position').
    - Detection of duplicate Image Index records.
    - Detection of empty / malformed / missing rows.
    - Identification of unknown finding labels.

    Args:
        csv_path: Path to the NIH annotations CSV file.
        output_json_path: Path to save the structured JSON artifact.
        output_csv_path: Path to save the backward-compatible CSV table.
        dataset_source_name: Documentation label for dataset provenance.

    Returns:
        Structured dictionary adhering to Module 8 specification.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If mandatory columns are missing or CSV is empty.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"NIH dataset annotations CSV not found at: {csv_file.resolve()}")

    seen_indices: Set[str] = set()
    duplicate_records_count = 0
    excluded_records_count = 0
    malformed_records_count = 0
    total_records_read = 0
    unknown_labels_found: Set[str] = set()

    # Per-disease frequency counters
    disease_counts: Dict[str, int] = {d: 0 for d in CANONICAL_DISEASES}
    no_finding_count = 0

    with open(csv_file, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file at {csv_file} is empty or lacks header row.")

        # Validate presence of mandatory columns
        fieldnames_normalized = {c.strip(): c for c in reader.fieldnames if c}
        missing_columns = [col for col in MANDATORY_CSV_COLUMNS if col not in fieldnames_normalized]
        if missing_columns:
            raise ValueError(
                f"Mandatory column(s) missing from CSV at {csv_file}: {missing_columns}. "
                f"Present columns: {list(reader.fieldnames)}"
            )

        img_col = fieldnames_normalized["Image Index"]
        lbl_col = fieldnames_normalized["Finding Labels"]

        for row in reader:
            total_records_read += 1
            if not row:
                malformed_records_count += 1
                continue

            img_id = (row.get(img_col) or "").strip()
            if not img_id:
                excluded_records_count += 1
                continue

            if img_id in seen_indices:
                duplicate_records_count += 1
                continue

            raw_labels = row.get(lbl_col)
            if raw_labels is None:
                malformed_records_count += 1
                continue

            seen_indices.add(img_id)
            labels_set = parse_labels_string(raw_labels)

            if NO_FINDING_LABEL in labels_set or "Normal" in labels_set:
                no_finding_count += 1

            for label in labels_set:
                if label in disease_counts:
                    disease_counts[label] += 1
                elif label not in (NO_FINDING_LABEL, "Normal"):
                    unknown_labels_found.add(label)

    total_valid_records = len(seen_indices)
    if total_valid_records == 0:
        raise ValueError(f"No valid unique image records found in CSV file at: {csv_file}")

    denominator = total_valid_records

    # Compute prevalences P_d = (F_d / N) * 100.0
    prevalences: Dict[str, float] = {}
    disease_records: List[Dict[str, Any]] = []

    for disease in CANONICAL_DISEASES:
        cnt = disease_counts[disease]
        prev = round((cnt / denominator) * 100.0, 4)
        prevalences[disease] = prev
        disease_records.append({
            "disease_name": disease,
            "occurrence_count": cnt,
            "prevalence_percentage": prev,
            "denominator": denominator,
            "dataset_source": dataset_source_name,
        })

    no_finding_prev = round((no_finding_count / denominator) * 100.0, 4)

    timestamp_iso = datetime.now(timezone.utc).isoformat()

    json_payload: Dict[str, Any] = {
        "dataset_name": "NIH ChestX-ray14",
        "dataset_source": dataset_source_name,
        "csv_path": str(csv_file.resolve()),
        "generated_at": timestamp_iso,
        "total_records_read": total_records_read,
        "total_valid_records": total_valid_records,
        "denominator": denominator,
        "duplicate_records_count": duplicate_records_count,
        "excluded_records_count": excluded_records_count,
        "malformed_records_count": malformed_records_count,
        "unknown_labels_found": sorted(list(unknown_labels_found)),
        "unknown_labels_count": len(unknown_labels_found),
        "diseases": disease_records,
        "disease_counts": disease_counts,
        "disease_prevalences": prevalences,
        "no_finding": {
            "disease_name": NO_FINDING_LABEL,
            "occurrence_count": no_finding_count,
            "prevalence_percentage": no_finding_prev,
            "denominator": denominator,
            "dataset_source": dataset_source_name,
        },
        "no_finding_count": no_finding_count,
        "no_finding_prevalence": no_finding_prev,
    }

    # Save to JSON
    if output_json_path:
        out_json = Path(output_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, mode="w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        json_payload["output_json_path"] = str(out_json.resolve())

    # Save to CSV for backward compatibility
    if output_csv_path:
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["disease_name", "image_count", "prevalence_percentage", "denominator", "dataset_source"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in disease_records:
                writer.writerow({
                    "disease_name": rec["disease_name"],
                    "image_count": rec["occurrence_count"],
                    "prevalence_percentage": rec["prevalence_percentage"],
                    "denominator": rec["denominator"],
                    "dataset_source": rec["dataset_source"],
                })
            # Separate No Finding row
            writer.writerow({
                "disease_name": NO_FINDING_LABEL,
                "image_count": no_finding_count,
                "prevalence_percentage": no_finding_prev,
                "denominator": denominator,
                "dataset_source": dataset_source_name,
            })
        json_payload["output_csv_path"] = str(out_csv.resolve())

    # Update in-memory caches only when writing to the default JSON artifact
    global _CACHED_JSON_DATA, _CACHED_CSV_TABLE
    if output_json_path is not None and Path(output_json_path).resolve() == DEFAULT_OUTPUT_JSON.resolve():
        _CACHED_JSON_DATA = json_payload
        _CACHED_CSV_TABLE = None

    return json_payload


def reset_disease_frequency_cache() -> None:
    """Reset module-level in-memory caches."""
    global _CACHED_JSON_DATA, _CACHED_CSV_TABLE
    _CACHED_JSON_DATA = None
    _CACHED_CSV_TABLE = None


def load_cached_disease_frequency_json(
    json_path: Union[str, Path] = DEFAULT_OUTPUT_JSON,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """
    Load precomputed disease frequency JSON artifact.
    Uses an in-memory cache for DEFAULT_OUTPUT_JSON to avoid re-reading disk on every query.
    If the JSON is missing, attempts to generate it from the master CSV.
    Fails clearly with FileNotFoundError if both JSON and CSV are unavailable.
    """
    global _CACHED_JSON_DATA
    jpath = Path(json_path).resolve()
    is_default = (jpath == DEFAULT_OUTPUT_JSON.resolve())

    if is_default and _CACHED_JSON_DATA is not None and not force_reload:
        return _CACHED_JSON_DATA

    jpath = Path(json_path)
    if not jpath.exists():
        if DEFAULT_NIH_CSV.exists():
            return compute_disease_frequencies(
                csv_path=DEFAULT_NIH_CSV,
                output_json_path=jpath,
            )
        raise FileNotFoundError(
            f"Disease frequency artifact not found at {jpath.resolve()} "
            f"and source CSV is unavailable at {DEFAULT_NIH_CSV.resolve()}."
        )

    with open(jpath, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate essential schema
    if (
        not isinstance(data, dict)
        or "total_valid_records" not in data
        or "disease_counts" not in data
        or "disease_prevalences" not in data
        or "no_finding" not in data
        or len(data.get("disease_counts", {})) < len(CANONICAL_DISEASES)
    ):
        raise ValueError(f"Corrupted or incomplete disease frequency JSON at: {jpath.resolve()}")

    _CACHED_JSON_DATA = data
    return _CACHED_JSON_DATA


def load_cached_disease_frequencies(
    freq_csv_path: Union[str, Path] = DEFAULT_OUTPUT_CSV,
) -> Dict[str, Dict[str, Any]]:
    """
    Load precomputed disease frequency table from CSV or JSON.
    Maintains backward compatibility with callers expecting dictionary keyed by disease name.
    """
    global _CACHED_CSV_TABLE
    if _CACHED_CSV_TABLE is not None:
        return _CACHED_CSV_TABLE

    fpath = Path(freq_csv_path)
    if not fpath.exists():
        if DEFAULT_OUTPUT_JSON.exists():
            json_data = load_cached_disease_frequency_json(DEFAULT_OUTPUT_JSON)
            table: Dict[str, Dict[str, Any]] = {}
            for rec in json_data["diseases"]:
                d_name = rec["disease_name"]
                table[d_name] = {
                    "disease_name": d_name,
                    "image_count": int(rec["occurrence_count"]),
                    "prevalence_percentage": float(rec["prevalence_percentage"]),
                    "denominator": int(rec["denominator"]),
                    "dataset_source": rec.get("dataset_source", "NIH ChestX-ray14"),
                }
            nf = json_data["no_finding"]
            table[NO_FINDING_LABEL] = {
                "disease_name": NO_FINDING_LABEL,
                "image_count": int(nf["occurrence_count"]),
                "prevalence_percentage": float(nf["prevalence_percentage"]),
                "denominator": int(nf["denominator"]),
                "dataset_source": nf.get("dataset_source", "NIH ChestX-ray14"),
            }
            _CACHED_CSV_TABLE = table
            return table

        compute_disease_frequencies(output_csv_path=fpath)

    table = {}
    with open(fpath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d_name = row["disease_name"]
            table[d_name] = {
                "disease_name": d_name,
                "image_count": int(row["image_count"]),
                "prevalence_percentage": float(row["prevalence_percentage"]),
                "denominator": int(row["denominator"]),
                "dataset_source": row.get("dataset_source", "NIH ChestX-ray14"),
            }
    _CACHED_CSV_TABLE = table
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute NIH ChestX-ray14 Disease Frequency & Prevalence")
    parser.add_argument("--csv", default=str(DEFAULT_NIH_CSV), help="Path to NIH Data_Entry CSV")
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON), help="Output JSON path")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV), help="Output CSV path")
    parser.add_argument("--regenerate", action="store_true", help="Force recalculation from CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("  COMPUTING NIH CHESTX-RAY14 GLOBAL DISEASE FREQUENCY TABLE")
    print("=" * 70)
    res = compute_disease_frequencies(
        csv_path=args.csv,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
    )
    print(f"Dataset Source       : {res['dataset_source']}")
    print(f"Total Valid Records  : {res['total_valid_records']:,}")
    print(f"Duplicates Excluded  : {res['duplicate_records_count']}")
    print(f"Malformed / Excluded : {res['excluded_records_count'] + res['malformed_records_count']}")
    print(f"Saved JSON Artifact  : {res['output_json_path']}")
    print(f"Saved CSV Artifact   : {res['output_csv_path']}")
    print("-" * 70)
    print(f"{'Disease Class':<24} {'Count':>10} {'Prevalence':>14}")
    print("-" * 70)
    for rec in res["diseases"]:
        print(f"{rec['disease_name']:<24} {rec['occurrence_count']:>10,} {rec['prevalence_percentage']:>13.4f}%")
    print("-" * 70)
    nf = res["no_finding"]
    print(f"{nf['disease_name']:<24} {nf['occurrence_count']:>10,} {nf['prevalence_percentage']:>13.4f}%")
    print("=" * 70)
