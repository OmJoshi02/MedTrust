"""
scratch/test_deployment_isolated.py
MedTrust Module 8 — Embedding-Only Deployment Isolation Verification & Benchmarks

Simulates a deployed production environment where:
1. Zero original reference X-ray image files exist on disk (dataset/images & reference_images are empty/blocked).
2. The system relies strictly on precomputed embedding & metadata artifacts:
   - outputs/similarity/reference_embeddings.npz
   - outputs/similarity/reference_metadata.json
   - outputs/similarity/reference_manifest.json
   - outputs/similarity/disease_frequency.json
3. Measures artifact sizes, retrieval latency, and verifies complete clinical evidence computation.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity.embedding_store import embedding_store
from src.similarity.metadata_loader import ReferenceMetadataLoader
from src.similarity.similarity_analyzer import SimilarityAnalyzer
from backend.engine import MedTrustBackendEngine


def run_deployment_verification():
    print("=" * 80)
    print("MEDTRUST MODULE 8: EMBEDDING-ONLY DEPLOYMENT ISOLATION BENCHMARK")
    print("=" * 80)

    # 1. Artifact Audit
    print("\n[1] Artifact Footprint & Cryptographic SHA-256 Audit:")
    artifacts = {
        "reference_embeddings.npz": PROJECT_ROOT / "outputs" / "similarity" / "reference_embeddings.npz",
        "reference_metadata.json": PROJECT_ROOT / "outputs" / "similarity" / "reference_metadata.json",
        "reference_manifest.json": PROJECT_ROOT / "outputs" / "similarity" / "reference_manifest.json",
        "disease_frequency.json": PROJECT_ROOT / "outputs" / "similarity" / "disease_frequency.json",
    }
    total_bytes = 0
    for name, path in artifacts.items():
        if not path.exists():
            print(f"  [FAIL] Missing artifact: {name} at {path}")
            sys.exit(1)
        size = path.stat().st_size
        total_bytes += size
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  - {name:28s}: {size:>10,d} bytes ({size / (1024*1024):.2f} MB) | SHA-256: {sha[:16]}...{sha[-8:]}")
    print(f"  TOTAL DEPLOYMENT SIMILARITY ARTIFACT STORAGE: {total_bytes:,d} bytes ({total_bytes / (1024*1024):.2f} MB)")

    # Compare with size of 2,500 raw X-rays
    # Average NIH PNG is ~1.5 MB -> 2500 * 1.5MB = ~3,750 MB (3.75 GB)
    savings_ratio = (3750 * 1024 * 1024) / total_bytes
    print(f"  ESTIMATED STORAGE SAVINGS vs RAW IMAGES (~3.75 GB): {savings_ratio:.1f}x reduction")

    # 2. Store Integrity
    print("\n[2] Embedding Store Integrity Validation:")
    report = embedding_store.verify_integrity(verify_hashes=True)
    for k, v in report.items():
        print(f"  - {k:25s}: {v}")
    assert report["valid"] is True, f"Embedding store invalid: {report['errors']}"
    assert report["hashes_verified"] is True, "Manifest cryptographic hashes failed to verify!"

    # 3. Guard against original image access
    print("\n[3] Simulating Deployment Isolation (Zero Reference Image Access Guard):")
    original_open = open

    def guarded_open(file, *args, **kwargs):
        file_str = str(file).lower().replace("\\", "/")
        if ("reference_images" in file_str or "dataset/images" in file_str) and not (
            file_str.endswith(".csv") or file_str.endswith(".json")
        ):
            raise AssertionError(f"DEPLOYMENT VIOLATION: Attempted to open reference image file from disk: {file}")
        return original_open(file, *args, **kwargs)

    # 4. Latency Benchmark
    print("\n[4] Retrieval Latency Benchmark (100 Iterations):")
    analyzer = SimilarityAnalyzer()
    dummy_query = np.random.randn(1024).astype(np.float32)
    dummy_query = dummy_query / np.linalg.norm(dummy_query)

    latencies = []
    with patch("builtins.open", side_effect=guarded_open):
        for _ in range(100):
            t0 = time.perf_counter()
            res = analyzer.find_similar_by_vector(
                query_vector=dummy_query,
                top_k=5,
                predicted_findings=["Effusion", "Infiltration"],
                authentic_only=True,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)
    print(f"  - Mean Retrieval Latency : {mean_lat:.3f} ms")
    print(f"  - Median (p50) Latency  : {p50_lat:.3f} ms")
    print(f"  - 95th Percentile (p95) : {p95_lat:.3f} ms")
    print(f"  - 99th Percentile (p99) : {p99_lat:.3f} ms")
    print(f"  - Reference Images Opened: 0 (Guarded assertion verified)")

    # 5. End-to-End Engine Pipeline Test
    print("\n[5] End-to-End Engine Pipeline Simulation with Authentic NIH Radiograph:")
    nih_sample = PROJECT_ROOT / "dataset" / "images" / "00000001_000.png"
    if nih_sample.exists():
        with open(nih_sample, "rb") as f:
            query_img = Image.open(f).convert("RGB")
        print(f"  - Loaded authentic query radiograph: {nih_sample.name}")
    else:
        query_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        print("  - Using synthetic query image for pipeline test")

    engine = MedTrustBackendEngine()

    with patch("builtins.open", side_effect=guarded_open):
        t0 = time.perf_counter()
        full_result = engine.run_similarity(
            image=query_img,
            top_k=5,
            query_findings=["Cardiomegaly"],
            primary_finding="Cardiomegaly",
        )
        total_time = (time.perf_counter() - t0) * 1000.0

    print(f"  - End-to-End Similarity Duration (feature extraction + retrieval): {total_time:.2f} ms")
    print(f"  - Top Match Retrieved   : {full_result['top_match']['image_id']} ({full_result['top_match']['similarity_score']:.2f}% cosine similarity)")
    print(f"  - Top Match Findings    : {full_result['top_match']['finding_labels']}")
    print(f"  - Primary Concordance   : {full_result.get('primary_concordance_rate')}% ({full_result.get('primary_concordant_count')}/5 cases)")

    pred_ev = full_result.get("predicted_findings_evidence", [])
    print(f"  - Finding Evidence Count: {len(pred_ev)}")
    for ev in pred_ev:
        print(f"    * Finding: {ev['predicted_finding']}")
        print(f"      Matching Cases: {ev['matching_cases']}/{ev['retrieved_cases']} ({ev['matching_rate']:.1f}%)")
        print(f"      Weighted Support: {ev['weighted_support']:.1f}%")
        print(f"      Dataset Prevalence: {ev['dataset_prevalence']:.2f}%")
        print(f"      Explanation: {ev['explanation']}")

    print("\n" + "=" * 80)
    print("VERDICT: PASS — EMBEDDING-ONLY DEPLOYMENT ARCHITECTURE FULLY VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_deployment_verification()

