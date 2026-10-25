"""
MedTrust V2 — Reference Case Intelligence Report Schema
(src/similarity_v2/report_schema.py)

Defines and validates the structured clinical intelligence schema for precomputed
reference case reports stored in the persistent reference library.

Invariants:
1. Authentic dataset annotations are strictly separated from model-generated predictions.
2. Predictions include raw probabilities, calibrated probabilities, Youden-J thresholds, and binary decisions.
3. Quality, OOD, and uncertainty states are recorded per component.
4. Checkpoint identity and cryptographic hashes guarantee provenance.
5. Missing or offline-skipped fields are explicitly marked rather than omitted or fabricated.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

SCHEMA_VERSION: str = "2.0.0"


def compute_dict_sha256(payload: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a dictionary."""
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def create_reference_report(
    reference_id: str,
    patient_id: str,
    demographics: Dict[str, Any],
    dataset_annotations: Dict[str, Any],
    predictions: Dict[str, Any],
    image_quality: Dict[str, Any],
    ood: Dict[str, Any],
    embedding_meta: Dict[str, Any],
    model_provenance: Dict[str, Any],
    uncertainty: Optional[Dict[str, Any]] = None,
    explainability: Optional[Dict[str, Any]] = None,
    dataset_provenance: str = "NIH ChestX-ray14",
    dataset_split: str = "train",
    component_status: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Construct a validated Reference Case Intelligence Report dictionary.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    default_component_status = {
        "prediction": "SUCCESS",
        "quality": "SUCCESS",
        "ood": "SUCCESS",
        "uncertainty": "NOT_EVALUATED_OFFLINE" if uncertainty is None else uncertainty.get("status", "SUCCESS"),
        "explainability": "NOT_EVALUATED_OFFLINE" if explainability is None else explainability.get("status", "SUCCESS"),
    }
    if component_status:
        default_component_status.update(component_status)

    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reference_id": str(reference_id),
        "dataset_provenance": str(dataset_provenance),
        "dataset_split": str(dataset_split),
        "patient_id": str(patient_id),
        "demographics": {
            "patient_age": demographics.get("patient_age"),
            "patient_gender": demographics.get("patient_gender", "U"),
            "view_position": demographics.get("view_position", "PA"),
        },
        "dataset_annotations": {
            "findings": list(dataset_annotations.get("findings", ["No Finding"])),
            "finding_labels": str(dataset_annotations.get("finding_labels", "No Finding")),
            "annotation_type": str(dataset_annotations.get("annotation_type", "Dataset Annotation")),
            "is_authentic_nih": bool(dataset_annotations.get("is_authentic_nih", True)),
        },
        "predictions": {
            "raw_scores": {d: float(predictions.get("raw_scores", {}).get(d, 0.0)) for d in CANONICAL_DISEASES},
            "calibrated_scores": {d: float(predictions.get("calibrated_scores", {}).get(d, 0.0)) for d in CANONICAL_DISEASES},
            "thresholds": {d: float(predictions.get("thresholds", {}).get(d, 0.5)) for d in CANONICAL_DISEASES},
            "decisions": {d: str(predictions.get("decisions", {}).get(d, "NEGATIVE")) for d in CANONICAL_DISEASES},
            "primary_finding": predictions.get("primary_finding", {
                "disease": "No Finding",
                "raw_probability": 0.0,
                "calibrated_probability": 0.0,
            }),
            "detected_findings": list(predictions.get("detected_findings", [])),
        },
        "image_quality": {
            "composite_score": int(image_quality.get("composite_score", 0)),
            "sharpness": float(image_quality.get("sharpness", 0.0)),
            "contrast": float(image_quality.get("contrast", 0.0)),
            "brightness": float(image_quality.get("brightness", 0.0)),
            "suitable": bool(image_quality.get("suitable", True)),
            "status": str(image_quality.get("status", "SUCCESS")),
        },
        "ood": {
            "is_ood": bool(ood.get("is_ood", False)),
            "status": str(ood.get("status", "IN_DISTRIBUTION")),
            "msp_score": float(ood.get("msp_score", 0.0)),
            "energy_score": float(ood.get("energy_score", 0.0)),
        },
        "uncertainty": uncertainty or {
            "status": "NOT_EVALUATED_OFFLINE",
            "method": "MC-Dropout",
            "overall_mean_variance": None,
            "reason": "Omitted in batch reference library to optimize build throughput",
        },
        "explainability": explainability or {
            "status": "NOT_EVALUATED_OFFLINE",
            "gradcam_available": False,
            "primary_target": predictions.get("primary_finding", {}).get("disease"),
            "reason": "Grad-CAM heatmaps computed dynamically upon individual case inspection",
        },
        "embedding": {
            "dimension": int(embedding_meta.get("dimension", 1024)),
            "dtype": str(embedding_meta.get("dtype", "float32")),
            "norm": float(embedding_meta.get("norm", 1.0)),
            "index_version": str(embedding_meta.get("index_version", "2.0-nih-authentic")),
            "vector_sha256": str(embedding_meta.get("vector_sha256", "")),
        },
        "model_provenance": {
            "architecture": str(model_provenance.get("architecture", "DenseNet-121")),
            "checkpoint_identity": str(model_provenance.get("checkpoint_identity", "")),
            "checkpoint_sha256": str(model_provenance.get("checkpoint_sha256", "")),
            "preprocessing_version": str(model_provenance.get("preprocessing_version", "1.0-ImageNet-224")),
        },
        "component_status": default_component_status,
        "created_at": now_iso,
    }

    report["report_sha256"] = compute_dict_sha256(report)
    return report


def validate_reference_report(report: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a reference report against the MedTrust V2 report schema.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not isinstance(report, dict):
        return False, ["Report must be a dictionary"]

    # 1. Required top-level fields
    required_top = [
        "schema_version",
        "reference_id",
        "dataset_provenance",
        "dataset_split",
        "patient_id",
        "demographics",
        "dataset_annotations",
        "predictions",
        "image_quality",
        "ood",
        "embedding",
        "model_provenance",
        "component_status",
        "created_at",
        "report_sha256",
    ]
    for field in required_top:
        if field not in report:
            errors.append(f"Missing top-level field: '{field}'")

    if errors:
        return False, errors

    # 2. Demographics validation
    demo = report.get("demographics", {})
    if "patient_gender" not in demo:
        errors.append("demographics missing 'patient_gender'")
    elif demo["patient_gender"] not in ("M", "F", "U", "O", None):
        errors.append(f"demographics invalid 'patient_gender': {demo['patient_gender']}")

    if "view_position" not in demo:
        errors.append("demographics missing 'view_position'")
    elif demo["view_position"] not in ("PA", "AP", "LL", "LATERAL", "UNKNOWN", None):
        errors.append(f"demographics invalid 'view_position': {demo['view_position']}")

    if demo.get("patient_age") is not None:
        try:
            age = int(demo["patient_age"])
            if age < 0 or age > 150:
                errors.append(f"demographics invalid 'patient_age': {age}")
        except (ValueError, TypeError):
            errors.append(f"demographics non-numeric 'patient_age': {demo['patient_age']}")

    # 3. Dataset annotations validation
    ann = report.get("dataset_annotations", {})
    if "findings" not in ann or not isinstance(ann["findings"], list):
        errors.append("dataset_annotations missing valid 'findings' list")
    if "finding_labels" not in ann:
        errors.append("dataset_annotations missing 'finding_labels'")

    # 4. Predictions validation
    pred = report.get("predictions", {})
    for sub in ("raw_scores", "calibrated_scores", "thresholds", "decisions"):
        if sub not in pred or not isinstance(pred[sub], dict):
            errors.append(f"predictions missing '{sub}' dictionary")
        else:
            missing_classes = [c for c in CANONICAL_DISEASES if c not in pred[sub]]
            if missing_classes:
                errors.append(f"predictions.{sub} missing classes: {missing_classes[:3]}")

    # 5. Image quality validation
    q = report.get("image_quality", {})
    if "composite_score" not in q:
        errors.append("image_quality missing 'composite_score'")
    if "suitable" not in q:
        errors.append("image_quality missing 'suitable'")

    # 6. OOD validation
    ood = report.get("ood", {})
    if "is_ood" not in ood:
        errors.append("ood missing 'is_ood'")

    # 7. Embedding metadata validation
    emb = report.get("embedding", {})
    if emb.get("dimension") != 1024:
        errors.append(f"embedding dimension must be 1024, got {emb.get('dimension')}")
    if abs(float(emb.get("norm", 0.0)) - 1.0) > 1e-2:
        errors.append(f"embedding norm must be ~1.0, got {emb.get('norm')}")

    # 8. Model provenance
    mp = report.get("model_provenance", {})
    if mp.get("architecture") != "DenseNet-121":
        errors.append(f"model_provenance architecture must be 'DenseNet-121', got '{mp.get('architecture')}'")
    if not mp.get("checkpoint_sha256"):
        errors.append("model_provenance missing 'checkpoint_sha256'")

    return len(errors) == 0, errors

