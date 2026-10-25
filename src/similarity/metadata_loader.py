"""
MedTrust V1 — Reference Cases Metadata Loader & Provenance Tracker
(src/similarity/metadata_loader.py)

Module 8: Similarity Analysis.
Loads, normalizes, and indexes authentic clinical metadata (dataset-provided
pathologies, demographics, radiographic view positions, and provenance)
for reference Chest X-rays from CSV catalogs.

Design & Compliance:
1. Primary Source: reference_images/reference_cases.csv
2. Fallback Source: dataset/Data_Entry_2017_v2020.csv (NIH master catalog)
3. Annotations are strictly identified as "Dataset Annotations", not clinical diagnoses.
4. Distinguishes authentic NIH ChestX-ray14 radiographs from synthetic demo images.
5. No fabricated clinical narrative notes or demographics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REF_CSV = PROJECT_ROOT / "reference_images" / "reference_cases.csv"
NIH_MASTER_CSV = PROJECT_ROOT / "dataset" / "Data_Entry_2017_v2020.csv"
DEFAULT_METADATA_JSON = PROJECT_ROOT / "outputs" / "similarity" / "reference_metadata.json"

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


def parse_finding_labels(raw_labels: Optional[str]) -> List[str]:
    """
    Parse pipe-separated or comma-separated finding labels into clean disease list.
    Preserves multiple disease labels and handles 'No Finding' / empty values.
    """
    if not raw_labels or str(raw_labels).strip() in ("", "None", "nan", "NaN"):
        return ["No Finding"]

    delimiters = ["|", ","]
    text = str(raw_labels).strip()
    for d in delimiters:
        if d in text:
            parts = [p.strip() for p in text.split(d) if p.strip()]
            return parts if parts else ["No Finding"]

    return [text]


class ReferenceMetadataLoader:
    """Loads and caches reference radiograph metadata from CSV catalogs or precomputed deployment JSON."""

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        nih_catalog_path: Optional[Union[str, Path]] = None,
        metadata_json_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.csv_path = Path(csv_path or DEFAULT_REF_CSV)
        self.nih_catalog_path = Path(nih_catalog_path or NIH_MASTER_CSV)
        self.metadata_json_path = Path(metadata_json_path or DEFAULT_METADATA_JSON)
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None
        self._nih_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_nih_catalog_loaded(self) -> Dict[str, Dict[str, Any]]:
        """Load and cache the full NIH master catalog into an in-memory dictionary for O(1) lookups."""
        if self._nih_cache is not None:
            return self._nih_cache

        nih_index: Dict[str, Dict[str, Any]] = {}
        if self.nih_catalog_path.exists():
            try:
                with open(self.nih_catalog_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        img_id = (row.get("Image Index") or "").strip()
                        if not img_id:
                            continue
                        raw_labels = row.get("Finding Labels", "No Finding")
                        findings = parse_finding_labels(raw_labels)
                        raw_age = row.get("Patient Age", "")
                        clean_age = "".join(c for c in str(raw_age) if c.isdigit())
                        age_val = int(clean_age) if clean_age else None
                        patient_id = (row.get("Patient ID") or "").strip()
                        nih_index[img_id] = {
                            "filename": img_id,
                            "image_index": img_id,
                            "patient_id": patient_id,
                            "finding_labels": raw_labels,
                            "findings": findings,
                            "patient_age": age_val,
                            "patient_gender": row.get("Patient Gender", "U"),
                            "view_position": row.get("View Position", "PA"),
                            "dataset_source": "NIH ChestX-ray14",
                            "annotation_type": "Dataset Annotation",
                        }
            except Exception as exc:
                print(f"[ReferenceMetadataLoader] Warning: Failed reading NIH catalog {self.nih_catalog_path}: {exc}")

        self._nih_cache = nih_index
        return self._nih_cache

    def load_index(self, force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
        """Load all metadata rows keyed by filename into memory."""
        if self._cache is not None and not force_reload:
            return self._cache

        index: Dict[str, Dict[str, Any]] = {}

        # 0. Check precomputed deployment JSON first (enables zero CSV/image dependency in deployment)
        if self.metadata_json_path.exists():
            try:
                import json
                with open(self.metadata_json_path, mode="r", encoding="utf-8") as f:
                    records = json.load(f)
                    if isinstance(records, list):
                        for rec in records:
                            fn = rec.get("filename") or rec.get("image_index")
                            if fn:
                                index[str(fn).strip()] = rec
            except Exception as exc:
                print(f"[ReferenceMetadataLoader] Warning: Failed reading {self.metadata_json_path}: {exc}")

        # 1. Load reference_cases.csv if available
        if self.csv_path.exists():
            try:
                with open(self.csv_path, mode="r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        filename = row.get("filename") or row.get("Image Index") or row.get("image_id")
                        if not filename:
                            continue
                        filename = filename.strip()
                        raw_labels = row.get("finding_labels") or row.get("Finding Labels") or "No Finding"
                        findings = parse_finding_labels(raw_labels)

                        age_val: Optional[Union[int, str]] = None
                        raw_age = row.get("patient_age") or row.get("Patient Age")
                        if raw_age and str(raw_age).strip().lower() not in ("unknown", "none", "nan", ""):
                            clean_age_str = "".join(c for c in str(raw_age) if c.isdigit())
                            age_val = int(clean_age_str) if clean_age_str else str(raw_age).strip()

                        gender = row.get("patient_gender") or row.get("Patient Gender") or "U"
                        view = row.get("view_position") or row.get("View Position") or "PA"
                        source = row.get("dataset_source") or ("NIH ChestX-ray14" if filename.endswith(".png") else "Synthetic Benchmark")
                        ann_type = row.get("annotation_type") or "Dataset Annotation"
                        img_idx = row.get("image_index") or filename
                        patient_id = (row.get("patient_id") or row.get("Patient ID") or "").strip()

                        index[filename] = {
                            "filename": filename,
                            "image_index": img_idx,
                            "patient_id": patient_id,
                            "finding_labels": str(raw_labels).strip(),
                            "findings": findings,
                            "patient_age": age_val,
                            "patient_gender": str(gender).strip(),
                            "view_position": str(view).strip(),
                            "dataset_source": source.strip(),
                            "annotation_type": ann_type.strip(),
                        }
            except Exception as exc:
                print(f"[ReferenceMetadataLoader] Warning: Failed reading {self.csv_path}: {exc}")

        self._cache = index
        return self._cache

    def get_metadata_for_filename(self, filename: str) -> Dict[str, Any]:
        """Retrieve metadata for a specific image filename, with fallback lookups."""
        index = self.load_index()
        basename = Path(filename).name

        if basename in index:
            return dict(index[basename])

        # 2. Check NIH master dataset if filename matches NIH format (e.g. 00000001_000.png)
        if self.nih_catalog_path.exists():
            nih_index = self._ensure_nih_catalog_loaded()
            if basename in nih_index:
                index[basename] = nih_index[basename]
                return dict(nih_index[basename])

        # 3. Fallback default metadata with clear provenance
        return {
            "filename": basename,
            "image_index": basename,
            "patient_id": "",
            "finding_labels": "Unspecified",
            "findings": ["Unspecified"],
            "patient_age": None,
            "patient_gender": "U",
            "view_position": "PA",
            "dataset_source": "Unspecified Source",
            "annotation_type": "Unspecified Annotation",
        }

    def get_metadata_list_for_filenames(self, filenames: List[str]) -> List[Dict[str, Any]]:
        """Return an ordered list of metadata dictionaries corresponding to each input filename."""
        return [self.get_metadata_for_filename(fn) for fn in filenames]


# Singleton default loader
metadata_loader = ReferenceMetadataLoader()
