"""
Independent Verification of Latency Calculations
Saves raw sample arrays, computes summary statistics independently,
and proves the cold-start vs warm-run mathematical mechanism.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity.feature_extractor import DenseNetSimilarityExtractor
from src.similarity_v2.retrieval_engine import SimilarityRetrievalEngineV2

# 1. Load 30 held-out validation queries
val_list_path = PROJECT_ROOT / "splits" / "val_list.txt"
images_dir = PROJECT_ROOT / "dataset" / "images"

with open(val_list_path, "r") as f:
    val_files = [line.strip() for line in f if line.strip()]

import random
rng = random.Random(42)
sampled_30 = rng.sample([f for f in val_files if (images_dir / f).exists()], 30)

# Pre-load images in memory to isolate model inference + retrieval latency
loaded_images = [(fn, Image.open(images_dir / fn).convert("RGB")) for fn in sampled_30]

print("=== PART A: UNWARMED COLD-START EXPERIMENT ===")
# Fresh cold extractor
cold_extractor = DenseNetSimilarityExtractor()
cold_engine = SimilarityRetrievalEngineV2(feature_extractor=cold_extractor)
cold_engine.index.load()

cold_samples = []
for idx, (fn, img) in enumerate(loaded_images):
    t0 = time.perf_counter()
    _ = cold_engine.find_similar_cases(img, top_k=5, exclude_filenames=[fn])
    dt_ms = (time.perf_counter() - t0) * 1000.0
    cold_samples.append(dt_ms)

cold_arr = np.array(cold_samples)
print("Raw cold samples (ms):")
print([round(x, 2) for x in cold_samples])
print(f"Sample count N: {len(cold_arr)}")
print(f"First query (cold spike): {cold_arr[0]:.2f} ms")
print(f"Mean:   {np.mean(cold_arr):.2f} ms")
print(f"Median: {np.median(cold_arr):.2f} ms")
print(f"P95:    {np.percentile(cold_arr, 95):.2f} ms")
print(f"Max:    {np.max(cold_arr):.2f} ms")
print(f"Min:    {np.min(cold_arr):.2f} ms")
print(f"Mean > P95?: {np.mean(cold_arr) > np.percentile(cold_arr, 95)}")

print("\n=== PART B: WARM BENCHMARK EXPERIMENT ===")
# 5 dummy warm-up passes
for i in range(5):
    _ = cold_engine.find_similar_cases(loaded_images[i][1], top_k=5)

warm_samples_by_k = {}
for k in [5, 10, 20]:
    k_samples = []
    for idx, (fn, img) in enumerate(loaded_images):
        t0 = time.perf_counter()
        _ = cold_engine.find_similar_cases(img, top_k=k, exclude_filenames=[fn])
        dt_ms = (time.perf_counter() - t0) * 1000.0
        k_samples.append(dt_ms)
    
    arr = np.array(k_samples)
    warm_samples_by_k[k] = {
        "raw_samples": [round(x, 3) for x in k_samples],
        "n": len(arr),
        "mean": round(float(np.mean(arr)), 2),
        "median": round(float(np.median(arr)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "max": round(float(np.max(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "mean_le_p95": bool(np.mean(arr) <= np.percentile(arr, 95))
    }
    print(f"K={k} Warm: Mean={warm_samples_by_k[k]['mean']}ms, Median={warm_samples_by_k[k]['median']}ms, P95={warm_samples_by_k[k]['p95']}ms, Max={warm_samples_by_k[k]['max']}ms (mean <= p95: {warm_samples_by_k[k]['mean_le_p95']})")

results = {
    "cold_run": {
        "raw_samples": [round(x, 3) for x in cold_samples],
        "n": len(cold_arr),
        "first_query_spike_ms": round(float(cold_arr[0]), 2),
        "mean_ms": round(float(np.mean(cold_arr)), 2),
        "median_ms": round(float(np.median(cold_arr)), 2),
        "p95_ms": round(float(np.percentile(cold_arr, 95)), 2),
        "max_ms": round(float(np.max(cold_arr)), 2),
        "min_ms": round(float(np.min(cold_arr)), 2),
        "mean_gt_p95": bool(np.mean(cold_arr) > np.percentile(cold_arr, 95))
    },
    "warm_runs": warm_samples_by_k
}

out_path = PROJECT_ROOT / "outputs" / "similarity_v2" / "evaluation_audit" / "verified_latency_audit.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nWrote verified latency results to {out_path}")

