"""
scratch/run_module8_acceptance.py
MedTrust Module 8 — Final Independent Deployment Acceptance Test Script

Executes rigorous checks for:
1. True image isolation (zero access to dataset/images/ or reference_images/)
2. Metadata-only loading & controlled error handling
3. Embedding compatibility & architecture invariants
4. Representative retrieval on genuine non-reference query radiograph
5. Cryptographic verification & regression health
"""

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity.embedding_store import EmbeddingStore, embedding_store
from src.similarity.metadata_loader import ReferenceMetadataLoader
from src.similarity.feature_extractor import DenseNetSimilarityExtractor
from src.similarity.disease_frequency import load_cached_disease_frequency_json
from src.similarity.similarity_analyzer import SimilarityAnalyzer
from backend.engine import MedTrustBackendEngine, verify_artifacts


def run_acceptance_tests():
    print("=" * 80)
    print("MEDTRUST MODULE 8: FINAL INDEPENDENT DEPLOYMENT ACCEPTANCE TEST")
    print("=" * 80)

    results = {}

    # --------------------------------------------------------------------------
    # CHECK 1: True Image Isolation
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CHECK 1: TRUE IMAGE ISOLATION")
    print("-" * 80)
    print("Verifying that query processing and Top-5 retrieval succeed with ZERO access")
    print("to original reference images or raw dataset image files...")

    # Load a genuine query image outside reference index (e.g. 00000001_000.png from val set)
    query_src = PROJECT_ROOT / "dataset" / "images" / "00000001_000.png"
    if not query_src.exists():
        print(f"ERROR: Sample query image {query_src} not found.")
        sys.exit(1)

    with open(query_src, "rb") as f:
        query_image_bytes = f.read()
    query_pil = Image.open(query_src).convert("RGB")
    print(f"Query image loaded: {query_src.name} ({len(query_image_bytes):,d} bytes)")

    # Install strict runtime guard: any open() targeting reference_images or dataset/images fails immediately
    original_open = open
    accessed_files = []

    def isolation_guarded_open(file, *args, **kwargs):
        normalized = str(file).lower().replace("\\", "/")
        if "dataset/images" in normalized or "reference_images" in normalized:
            if not (normalized.endswith(".csv") or normalized.endswith(".json")):
                accessed_files.append(normalized)
                raise AssertionError(
                    f"DEPLOYMENT ISOLATION VIOLATION: Tried to open raw image from disk: {file}"
                )
        return original_open(file, *args, **kwargs)

    try:
        with patch("builtins.open", side_effect=isolation_guarded_open):
            engine = MedTrustBackendEngine()
            t0 = time.perf_counter()
            sim_output = engine.run_similarity(
                image=query_pil,
                top_k=5,
                query_findings=["Cardiomegaly"],
                primary_finding="Cardiomegaly",
                authentic_only=True,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(accessed_files) == 0, f"Raw image files were accessed: {accessed_files}"
        assert sim_output is not None, "Engine returned None"
        assert "matches" in sim_output, "No 'matches' key in similarity output"
        assert len(sim_output["matches"]) == 5, f"Expected 5 matches, got {len(sim_output['matches'])}"
        assert "predicted_findings_evidence" in sim_output, "Missing predicted_findings_evidence"

        print(f"  [OK] Zero reference images accessed (violation count: {len(accessed_files)})")
        print(f"  [OK] Top-5 retrieval completed in {elapsed_ms:.2f} ms")
        print(f"  [OK] Top match: {sim_output['top_match']['image_id']} (similarity: {sim_output['top_match']['similarity_score']:.2f}%)")
        print(f"  [OK] Evidence: {sim_output['predicted_findings_evidence'][0]['explanation']}")
        results["check_1_image_isolation"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] Check 1 failed: {exc}")
        results["check_1_image_isolation"] = f"FAIL: {exc}"

    # --------------------------------------------------------------------------
    # CHECK 2: Metadata-Only Loading & Controlled Error Handling
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CHECK 2: METADATA-ONLY LOADING & CONTROLLED ERROR HANDLING")
    print("-" * 80)
    try:
        # A. Verify loading solely from reference_metadata.json (no CSVs)
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy_ref_csv = Path(tmpdir) / "nonexistent_ref.csv"
            dummy_nih_csv = Path(tmpdir) / "nonexistent_nih.csv"
            loader = ReferenceMetadataLoader(
                csv_path=dummy_ref_csv,
                nih_catalog_path=dummy_nih_csv,
            )
            index = loader.load_index()
            assert len(index) == 2500, f"Expected 2500 metadata records, got {len(index)}"
            print(f"  [OK] ReferenceMetadataLoader loaded {len(index)} records with CSVs completely absent")

        # B. Verify controlled error on missing embeddings
        store_missing = EmbeddingStore(PROJECT_ROOT / "outputs" / "similarity" / "nonexistent_embeddings.npz")
        try:
            store_missing.load()
            raise AssertionError("Failed to raise FileNotFoundError for missing embeddings")
        except FileNotFoundError as fnf_err:
            print(f"  [OK] Controlled FileNotFoundError on missing embeddings: {fnf_err}")

        # C. Verify controlled error on corrupted disease frequency JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_json = Path(tmpdir) / "corrupt_freq.json"
            corrupt_json.write_text('{"bad_key": 123}', encoding="utf-8")
            try:
                load_cached_disease_frequency_json(corrupt_json)
                raise AssertionError("Failed to raise ValueError for corrupt disease frequency JSON")
            except ValueError as val_err:
                print(f"  [OK] Controlled ValueError on corrupted disease frequency: {val_err}")

        results["check_2_metadata_and_errors"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] Check 2 failed: {exc}")
        results["check_2_metadata_and_errors"] = f"FAIL: {exc}"

    # --------------------------------------------------------------------------
    # CHECK 3: Embedding Compatibility & Architecture Invariants
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CHECK 3: EMBEDDING COMPATIBILITY & ARCHITECTURE INVARIANTS")
    print("-" * 80)
    try:
        # A. Manifest audit
        manifest_path = PROJECT_ROOT / "outputs" / "similarity" / "reference_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"  - Model architecture : {manifest.get('model_architecture')}")
        print(f"  - Feature dimension  : {manifest.get('feature_dimension')}")
        print(f"  - Checkpoint identity: {manifest.get('checkpoint_identity')}")
        print(f"  - Normalization      : {manifest.get('normalization')}")
        print(f"  - Total references   : {manifest.get('total_reference_images')}")
        print(f"  - Requires images    : {manifest.get('requires_original_images')}")

        assert manifest.get("model_architecture") == "DenseNet-121"
        assert manifest.get("feature_dimension") == 1024
        assert manifest.get("total_reference_images") == 2500
        assert manifest.get("requires_original_images") is False

        # B. Checkpoint SHA-256 verification
        ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pt"
        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        assert ckpt_sha == manifest.get("checkpoint_sha256"), (
            f"Checkpoint hash mismatch: {ckpt_sha} vs {manifest.get('checkpoint_sha256')}"
        )
        print(f"  [OK] Checkpoint SHA-256 matches manifest: {ckpt_sha}")

        # C. Query extractor compatibility
        extractor = DenseNetSimilarityExtractor()
        query_vec = extractor.extract_features(query_pil, normalize=True)
        assert query_vec.shape == (1024,), f"Expected shape (1024,), got {query_vec.shape}"
        assert query_vec.dtype == np.float32, f"Expected float32, got {query_vec.dtype}"
        assert np.all(np.isfinite(query_vec)), "Query vector contains non-finite values"
        norm = np.linalg.norm(query_vec)
        assert abs(norm - 1.0) < 1e-3, f"Query vector not unit normalized: {norm}"
        print(f"  [OK] Query vector is 1024-d, float32, finite, unit-norm (norm = {norm:.6f})")

        # D. Reference store integrity
        integrity = embedding_store.verify_integrity(verify_hashes=True)
        assert integrity["valid"] is True, f"Integrity errors: {integrity['errors']}"
        assert integrity["count"] == 2500
        assert integrity["dimension"] == 1024
        assert integrity["hashes_verified"] is True
        print(f"  [OK] Reference index verified: 2500 vectors, 1024-d, unit-norm, hashes verified")

        results["check_3_embedding_compatibility"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] Check 3 failed: {exc}")
        results["check_3_embedding_compatibility"] = f"FAIL: {exc}"

    # --------------------------------------------------------------------------
    # CHECK 4: Representative Retrieval on Genuine Non-Reference Radiograph
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CHECK 4: REPRESENTATIVE RETRIEVAL (NON-REFERENCE QUERY)")
    print("-" * 80)
    try:
        analyzer = SimilarityAnalyzer()
        query_fn = "00000001_000.png"

        # Assert query is definitely NOT in reference index
        ref_filenames, _ = embedding_store.get_data()
        assert query_fn not in ref_filenames, f"Query {query_fn} should not be in reference set!"

        retrieval_res = analyzer.find_similar_by_vector(
            query_vector=query_vec,
            top_k=5,
            predicted_findings=["Cardiomegaly"],
            primary_finding="Cardiomegaly",
            authentic_only=True,
        )

        matches = retrieval_res["matches"]
        assert len(matches) == 5, f"Expected 5 matches, got {len(matches)}"

        print(f"Retrieved Top-5 reference matches for non-reference query '{query_fn}':")
        for m in matches:
            assert m["similarity_score"] >= 0.0 and m["similarity_score"] <= 100.0
            assert "findings" in m
            assert "metadata" in m
            print(f"  Rank {m['rank']}: {m['image_id']:16s} | Similarity: {m['similarity_score']:>5.2f}% | Findings: {m['finding_labels']:30s} | {m['metadata'].get('patient_age')}y {m['metadata'].get('patient_gender')} {m['metadata'].get('view_position')}")

        # Check finding evidence
        ev = retrieval_res["predicted_findings_evidence"][0]
        print(f"\nPrediction-Driven Evidence: {ev['explanation']}")
        print(f"  - Finding: {ev['predicted_finding']}")
        print(f"  - Matching Cases: {ev['matching_cases']} of {ev['retrieved_cases']} ({ev['matching_rate']:.1f}%)")
        print(f"  - Weighted Support: {ev['weighted_support']:.1f}%")
        print(f"  - Global Prevalence: {ev['dataset_prevalence']:.2f}%")

        # Verify disclaimer presence (decision-support, not diagnostic claim)
        assert "disclaimer" in retrieval_res
        print(f"  [OK] Standard decision-support disclaimer present: {retrieval_res['disclaimer'][:80]}...")

        results["check_4_representative_retrieval"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] Check 4 failed: {exc}")
        results["check_4_representative_retrieval"] = f"FAIL: {exc}"

    # --------------------------------------------------------------------------
    # CHECK 5: Regression & Security Verification
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("CHECK 5: SYSTEM ARTIFACT INTEGRITY AUDIT")
    print("-" * 80)
    try:
        art_report = verify_artifacts(PROJECT_ROOT)
        print(f"  - Artifact report ready       : {art_report['ready']}")
        print(f"  - Critical artifacts present  : {art_report['critical_present']}")
        print(f"  - Total failures              : {len(art_report['failures'])}")
        assert art_report["ready"] is True
        assert len(art_report["failures"]) == 0, f"Artifact failures: {art_report['failures']}"
        print("  [OK] All system artifacts (weights, thresholds, temperatures, embeddings) cryptographically verified")
        results["check_5_artifact_integrity"] = "PASS"
    except Exception as exc:
        print(f"  [FAIL] Check 5 failed: {exc}")
        results["check_5_artifact_integrity"] = f"FAIL: {exc}"

    # --------------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ACCEPTANCE SUMMARY RESULTS:")
    print("=" * 80)
    all_pass = True
    for k, v in results.items():
        status = "PASS" if v == "PASS" else "FAIL"
        print(f"  {k:35s}: {status} ({v})")
        if v != "PASS":
            all_pass = False

    if all_pass:
        print("\nFINAL OVERALL ACCEPTANCE VERDICT: PASS")
    else:
        print("\nFINAL OVERALL ACCEPTANCE VERDICT: FAIL")

    return all_pass


if __name__ == "__main__":
    success = run_acceptance_tests()
    sys.exit(0 if success else 1)

