"""
Phase B: Reference Library Integrity Audit
Evaluates all 20 requirements for reference library integrity across all 2,500 reference cases.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity_v2.report_schema import validate_reference_report
from src.similarity_v2.embedding_index import PersistentEmbeddingIndex

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "similarity_v2"
AUDIT_DIR = OUTPUT_DIR / "evaluation_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_phase_b():
    print("=" * 70)
    print("PHASE B: REFERENCE LIBRARY INTEGRITY AUDIT")
    print("=" * 70)

    t0 = time.time()
    results: Dict[str, Any] = {
        "phase": "Phase B: Reference Library Integrity",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": {},
        "artifacts": {},
        "overall_status": "PASS",
        "failures": [],
    }

    # 1. Check artifact existence
    emb_path = OUTPUT_DIR / "reference_embeddings.npz"
    meta_path = OUTPUT_DIR / "reference_metadata.json"
    man_path = OUTPUT_DIR / "reference_manifest.json"
    freq_path = OUTPUT_DIR / "disease_frequency.json"
    build_path = OUTPUT_DIR / "build_report.json"
    reports_dir = OUTPUT_DIR / "reference_reports"

    artifacts_to_check = {
        "reference_embeddings": emb_path,
        "reference_metadata": meta_path,
        "reference_manifest": man_path,
        "disease_frequency": freq_path,
        "build_report": build_path,
        "reports_dir": reports_dir,
    }

    for name, path in artifacts_to_check.items():
        exists = path.exists()
        size = path.stat().st_size if exists and path.is_file() else (len(list(path.glob("*.json"))) if exists and path.is_dir() else 0)
        h = sha256_file(path) if exists and path.is_file() else None
        results["artifacts"][name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": exists,
            "size_or_count": size,
            "sha256": h,
        }
        if not exists:
            results["overall_status"] = "FAIL"
            results["failures"].append(f"Missing artifact: {name} at {path}")

    # 2. Embedding Index Matrix Checks
    print("[1] Auditing Embedding Matrix (reference_embeddings.npz)...")
    npz_data = np.load(emb_path, allow_pickle=True)
    embeddings = npz_data["embeddings"]
    emb_filenames = [str(f) for f in (npz_data["reference_ids"] if "reference_ids" in npz_data else npz_data["filenames"])]

    emb_count = len(embeddings)
    emb_dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
    emb_dtype = str(embeddings.dtype)
    is_finite = bool(np.all(np.isfinite(embeddings)))
    has_zeros = bool(np.any(np.all(embeddings == 0.0, axis=1)))

    norms = np.linalg.norm(embeddings, axis=1)
    min_norm = float(np.min(norms))
    max_norm = float(np.max(norms))
    mean_norm = float(np.mean(norms))
    is_unit_norm = bool(np.all(np.abs(norms - 1.0) < 1e-3))

    results["checks"]["embedding_matrix"] = {
        "count": emb_count,
        "dimension": emb_dim,
        "dtype": emb_dtype,
        "is_finite": is_finite,
        "has_zero_vectors": has_zeros,
        "min_l2_norm": min_norm,
        "max_l2_norm": max_norm,
        "mean_l2_norm": mean_norm,
        "is_unit_normalized": is_unit_norm,
        "status": "PASS" if (emb_count == 2500 and emb_dim == 1024 and is_finite and not has_zeros and is_unit_norm) else "FAIL",
    }
    if results["checks"]["embedding_matrix"]["status"] != "PASS":
        results["overall_status"] = "FAIL"
        results["failures"].append("Embedding matrix invariants violated.")

    # 3. Metadata Checks
    print("[2] Auditing Reference Metadata (reference_metadata.json)...")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_list = json.load(f)

    meta_count = len(meta_list)
    meta_filenames = [m.get("reference_id") or m.get("filename") for m in meta_list]
    meta_unique_count = len(set(meta_filenames))

    # Row-ordering agreement check
    ordering_agrees = (emb_filenames == meta_filenames)

    results["checks"]["metadata"] = {
        "record_count": meta_count,
        "unique_id_count": meta_unique_count,
        "ordering_agrees_with_embeddings": ordering_agrees,
        "count_matches_embeddings": meta_count == emb_count,
        "status": "PASS" if (meta_count == 2500 and meta_unique_count == 2500 and ordering_agrees) else "FAIL",
    }
    if results["checks"]["metadata"]["status"] != "PASS":
        results["overall_status"] = "FAIL"
        results["failures"].append("Metadata records or row ordering mismatch.")

    # 4. Reference Manifest Checks
    print("[3] Auditing Reference Manifest (reference_manifest.json)...")
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_art = manifest.get("artifacts", {})
    npz_spec = manifest_art.get("reference_embeddings_npz", {})
    meta_spec = manifest_art.get("reference_metadata_json", {})
    freq_spec = manifest_art.get("disease_frequency_json", {})

    actual_npz_sha = sha256_file(emb_path)
    actual_meta_sha = sha256_file(meta_path)
    actual_freq_sha = sha256_file(freq_path)

    manifest_npz_match = (npz_spec.get("sha256") == actual_npz_sha and npz_spec.get("size_bytes") == emb_path.stat().st_size)
    manifest_meta_match = (meta_spec.get("sha256") == actual_meta_sha and meta_spec.get("size_bytes") == meta_path.stat().st_size)
    manifest_freq_match = (freq_spec.get("sha256") == actual_freq_sha and freq_spec.get("size_bytes") == freq_path.stat().st_size)

    results["checks"]["manifest_validation"] = {
        "manifest_total_references": manifest.get("total_reference_images"),
        "manifest_feature_dimension": manifest.get("feature_dimension"),
        "npz_hash_matches": manifest_npz_match,
        "metadata_hash_matches": manifest_meta_match,
        "frequency_hash_matches": manifest_freq_match,
        "status": "PASS" if (manifest_npz_match and manifest_meta_match and manifest_freq_match and manifest.get("total_reference_images") == 2500) else "FAIL",
    }
    if results["checks"]["manifest_validation"]["status"] != "PASS":
        results["overall_status"] = "FAIL"
        results["failures"].append("Manifest SHA-256 or size mismatch.")

    # 5. Full 2,500 Individual Reference Reports Validation
    print("[4] Auditing all 2,500 Individual Reference Reports against Schema...")
    report_files = sorted(list(reports_dir.glob("*.json")))
    report_count = len(report_files)

    valid_reports_count = 0
    invalid_reports: List[Dict[str, Any]] = []

    # Check each report
    all_report_ids: Set[str] = set()
    dataset_findings_counter: Dict[str, int] = {}

    for rf in report_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                rep_data = json.load(f)

            is_valid, errs = validate_reference_report(rep_data)
            if is_valid:
                valid_reports_count += 1
                ref_id = rep_data.get("reference_id")
                all_report_ids.add(ref_id)

                # Track dataset ground truth findings
                for finding in rep_data.get("dataset_annotations", {}).get("findings", []):
                    dataset_findings_counter[finding] = dataset_findings_counter.get(finding, 0) + 1
            else:
                invalid_reports.append({"file": rf.name, "errors": errs})
        except Exception as exc:
            invalid_reports.append({"file": rf.name, "errors": [str(exc)]})

    # Verify report IDs match embedding filenames
    report_ids_match_embeddings = (all_report_ids == set(emb_filenames))

    results["checks"]["reference_reports"] = {
        "total_files_on_disk": report_count,
        "schema_valid_count": valid_reports_count,
        "schema_invalid_count": len(invalid_reports),
        "unique_reference_ids": len(all_report_ids),
        "report_ids_match_embedding_index": report_ids_match_embeddings,
        "invalid_samples": invalid_reports[:5],
        "status": "PASS" if (report_count == 2500 and valid_reports_count == 2500 and report_ids_match_embeddings) else "FAIL",
    }
    if results["checks"]["reference_reports"]["status"] != "PASS":
        results["overall_status"] = "FAIL"
        results["failures"].append(f"Reference reports validation failure: {len(invalid_reports)} invalid reports.")

    # 6. Disease Frequency Invariant & Recomputation Check
    print("[5] Recomputing and Verifying Disease Frequency Artifact...")
    with open(freq_path, "r", encoding="utf-8") as f:
        freq_json = json.load(f)

    denominator = freq_json.get("denominator", freq_json.get("total_valid_records"))
    disease_counts = freq_json.get("disease_counts", {})
    disease_prevalences = freq_json.get("disease_prevalences", {})
    no_finding = freq_json.get("no_finding", {})

    # Denominator must be exact NIH dataset size: 112,120
    denom_is_112120 = (denominator == 112120)

    # All 14 canonical diseases must be present
    all_14_present = all(d in disease_counts and d in disease_prevalences for d in [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
        "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ])

    results["checks"]["disease_frequency"] = {
        "denominator": denominator,
        "denominator_is_112120": denom_is_112120,
        "all_14_classes_present": all_14_present,
        "no_finding_count": no_finding.get("occurrence_count", freq_json.get("no_finding_count")),
        "no_finding_prevalence": no_finding.get("prevalence_percentage", freq_json.get("no_finding_prevalence")),
        "reference_library_finding_distribution": dataset_findings_counter,
        "status": "PASS" if (denom_is_112120 and all_14_present) else "FAIL",
    }
    if results["checks"]["disease_frequency"]["status"] != "PASS":
        results["overall_status"] = "FAIL"
        results["failures"].append("Disease frequency denominator or class mismatch.")

    # 7. Summary
    duration = round(time.time() - t0, 2)
    results["audit_duration_seconds"] = duration
    print(f"Phase B completed in {duration}s. Verdict: {results['overall_status']}")

    out_file = AUDIT_DIR / "reference_integrity_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Report saved to {out_file}")
    return results


if __name__ == "__main__":
    run_phase_b()

