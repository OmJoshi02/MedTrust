#!/usr/bin/env python3
"""
MedTrust V1 — Production NIH Reference Embedding Generator (scripts/generate_reference_embeddings.py)

Module 8: Similarity Analysis.
Precomputes 1024-dimensional DenseNet-121 feature vectors for eligible authentic
NIH ChestX-ray14 reference images from the training cohort (splits/train_list.txt),
and saves them to outputs/similarity/reference_embeddings.npz with comprehensive
clinical metadata and provenance manifest.

Methodological & Scientific Invariants (AGENTS.md Rules 4, 5, 18 & Specification):
1. Eligible Reference Population: Strictly derived from splits/train_list.txt (Zero Test/Val Leakage).
2. Ground-Truth Source: Dataset annotations joined directly from dataset/Data_Entry_2017_v2020.csv.
3. Feature Representation: DenseNet-121 backbone (features -> relu -> adaptive_avg_pool2d(1, 1) -> 1024-d).
4. Normalization: L2 unit-norm (||v||_2 = 1.0) with non-finite / NaN checks.
5. Zero Synthetic Cases: Authentic NIH ChestX-ray14 radiographs only (no demo images in research index).
6. Atomic Saving: Prevents partial / corrupted index generation.
7. Truthful Documentation: Manifest records exact indexed image count, source split, and model fingerprint.

Usage:
    python scripts/generate_reference_embeddings.py [--max-images 2500] [--device cuda] [--batch-size 64]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.similarity.feature_extractor import DenseNetSimilarityExtractor
from src.similarity.metadata_loader import ReferenceMetadataLoader, parse_finding_labels
from src.similarity.disease_frequency import CANONICAL_DISEASES, NO_FINDING_LABEL

DEFAULT_SPLIT_FILE = PROJECT_ROOT / "splits" / "train_list.txt"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
DEFAULT_CSV_PATH = PROJECT_ROOT / "dataset" / "Data_Entry_2017_v2020.csv"
DEFAULT_OUTPUT_NPZ = PROJECT_ROOT / "outputs" / "similarity" / "reference_embeddings.npz"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "outputs" / "similarity" / "reference_manifest.json"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "outputs" / "similarity" / "reference_metadata.json"

EXCLUDED_QUERY_FILENAMES = {"sample_query.jpg", "sample_query.png", "disease.png", "disease2.jpg"}


def select_stratified_training_cohort(
    split_file: Path,
    csv_path: Path,
    images_dir: Path,
    max_images: Optional[int] = 2500,
    seed: int = 42,
) -> List[str]:
    """
    Select an authentic, balanced, and diverse reference cohort from splits/train_list.txt.
    Guarantees representation across all 14 canonical disease classes and No Finding.
    """
    if not split_file.exists():
        raise FileNotFoundError(f"Training split file not found: {split_file}")
    if not csv_path.exists():
        raise FileNotFoundError(f"NIH master CSV catalog not found: {csv_path}")

    with open(split_file, "r", encoding="utf-8") as f:
        eligible_set = {line.strip() for line in f if line.strip()}

    # Exclude test and val lists explicitly to guarantee zero leakage
    for forbidden in ["val_list.txt", "test_list.txt"]:
        fpath = split_file.parent / forbidden
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                forbidden_set = {line.strip() for line in f if line.strip()}
                eligible_set -= forbidden_set

    for q in EXCLUDED_QUERY_FILENAMES:
        eligible_set.discard(q)

    # Group eligible images by disease category
    category_to_imgs: Dict[str, List[str]] = defaultdict(list)
    img_to_findings: Dict[str, List[str]] = {}

    with open(csv_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = (row.get("Image Index") or "").strip()
            if img_id in eligible_set and (images_dir / img_id).exists():
                raw_labels = row.get("Finding Labels", NO_FINDING_LABEL)
                findings = parse_finding_labels(raw_labels)
                img_to_findings[img_id] = findings
                for f_name in findings:
                    category_to_imgs[f_name].append(img_id)

    rng = random.Random(seed)

    if max_images is None or max_images >= len(img_to_findings):
        # Index all available training images
        return sorted(list(img_to_findings.keys()))

    # Stratified selection to ensure balanced coverage across all 14 diseases + No Finding
    selected_set: Set[str] = set()

    # 1. First priority: include all available cases for rare diseases (e.g. Hernia)
    for disease in CANONICAL_DISEASES:
        available = category_to_imgs.get(disease, [])
        if len(available) <= 200:
            selected_set.update(available)

    # 2. Balanced sampling across other categories
    target_per_disease = max(50, (max_images - len(selected_set)) // (len(CANONICAL_DISEASES) + 1))
    for disease in list(CANONICAL_DISEASES) + [NO_FINDING_LABEL]:
        available = [img for img in category_to_imgs.get(disease, []) if img not in selected_set]
        rng.shuffle(available)
        quota = min(target_per_disease, len(available))
        selected_set.update(available[:quota])

    # 3. Fill remaining quota from remaining eligible training images
    remaining = [img for img in img_to_findings.keys() if img not in selected_set]
    rng.shuffle(remaining)
    needed = max_images - len(selected_set)
    if needed > 0:
        selected_set.update(remaining[:needed])

    selected_list = sorted(list(selected_set))
    return selected_list[:max_images]


def generate_reference_embeddings(
    split_file: Path = DEFAULT_SPLIT_FILE,
    images_dir: Path = DEFAULT_IMAGES_DIR,
    csv_path: Path = DEFAULT_CSV_PATH,
    output_path: Path = DEFAULT_OUTPUT_NPZ,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    max_images: Optional[int] = 2500,
    batch_size: int = 64,
    device_str: str = "cuda",
    seed: int = 42,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  MEDTRUST V1 — PRODUCTION NIH REFERENCE EMBEDDING GENERATOR (MODULE 8)")
    print("=" * 75)
    print(f"Training split file  : {split_file}")
    print(f"Dataset images dir   : {images_dir}")
    print(f"NIH master CSV       : {csv_path}")
    print(f"Output archive path  : {output_path}")
    print(f"Target sample size   : {max_images if max_images else 'All Training Images'}")
    print(f"Compute device       : {device_str}")
    print(f"Batch size           : {batch_size}")

    print("\n[1/4] Selecting stratified authentic NIH training cohort...")
    selected_filenames = select_stratified_training_cohort(
        split_file=split_file,
        csv_path=csv_path,
        images_dir=images_dir,
        max_images=max_images,
        seed=seed,
    )
    n_images = len(selected_filenames)
    print(f"      Selected {n_images:,} authentic NIH training images for reference index.")

    print("\n[2/4] Joining verified dataset metadata from NIH CSV catalog...")
    meta_loader = ReferenceMetadataLoader(nih_catalog_path=csv_path)
    meta_loader._ensure_nih_catalog_loaded()
    metadata_list = [meta_loader.get_metadata_for_filename(fn) for fn in selected_filenames]

    # Verify label distribution in selected cohort
    cohort_disease_counts: Dict[str, int] = defaultdict(int)
    for meta in metadata_list:
        meta["dataset_split"] = "train"
        for f in meta.get("findings", []):
            cohort_disease_counts[f] += 1

    print("      Reference cohort finding distribution:")
    for d in CANONICAL_DISEASES:
        print(f"        - {d:<20}: {cohort_disease_counts[d]:4d} cases")
    print(f"        - {NO_FINDING_LABEL:<20}: {cohort_disease_counts[NO_FINDING_LABEL]:4d} cases")

    print("\n[3/4] Initializing DenseNet-121 similarity feature extractor...")
    import torch
    dev = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    extractor = DenseNetSimilarityExtractor(device=dev)
    manifest = extractor.get_provenance_manifest()
    print(f"      Model Architecture   : {manifest['model_architecture']}")
    print(f"      Checkpoint Identity  : {manifest['checkpoint_identity']}")
    if manifest["checkpoint_sha256"]:
        print(f"      Checkpoint SHA-256   : {manifest['checkpoint_sha256']}")
    print(f"      Feature Dimension    : {manifest['feature_dimension']}")

    print(f"\n[4/4] Extracting {n_images:,} embeddings in batches of {batch_size}...")
    start_time = time.time()
    image_paths = [images_dir / fn for fn in selected_filenames]

    # Batch extraction
    embeddings_matrix = extractor.extract_batch(
        image_inputs=image_paths,
        normalize=True,
        batch_size=batch_size,
    )

    elapsed = time.time() - start_time
    fps = n_images / elapsed if elapsed > 0 else 0
    print(f"      Extracted {embeddings_matrix.shape[0]:,} vectors in {elapsed:.2f}s ({fps:.1f} imgs/sec).")

    # Final integrity validation
    if embeddings_matrix.shape != (n_images, 1024):
        raise ValueError(f"Shape mismatch: expected ({n_images}, 1024), got {embeddings_matrix.shape}")
    if not np.all(np.isfinite(embeddings_matrix)):
        raise ValueError("Non-finite values (NaN/Inf) detected in extracted embeddings.")

    # Manifest metadata
    manifest.update({
        "total_reference_images": n_images,
        "source_dataset": "NIH ChestX-ray14",
        "source_split": str(split_file.name),
        "selection_strategy": "stratified_balanced_training_cohort",
        "random_seed": seed,
        "synthetic_cases_count": 0,
        "authentic_nih_cases_count": n_images,
        "embedding_shape": list(embeddings_matrix.shape),
        "embedding_dtype": str(embeddings_matrix.dtype),
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "build_duration_seconds": round(elapsed, 2),
        "index_version": "2.0-nih-authentic",
    })
    manifest_json_str = json.dumps(manifest, indent=2)

    # Atomic write for .npz archive
    tmp_output = output_path.with_suffix(".tmp.npz")
    print(f"\nWriting compressed archive atomically to {output_path}...")
    np.savez_compressed(
        tmp_output,
        filenames=np.array(selected_filenames, dtype=str),
        embeddings=embeddings_matrix,
        metadata=metadata_list,
        manifest=manifest_json_str,
    )
    if output_path.exists():
        output_path.unlink()
    tmp_output.rename(output_path)

    # Write companion manifest and metadata files
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_json_str)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print("=" * 75)
    print("  NIH REFERENCE EMBEDDINGS INDEX GENERATED SUCCESSFULLY")
    print(f"  Indexed Images         : {n_images:,} (100% authentic NIH training cases)")
    print(f"  Synthetic Images       : 0 (strictly excluded)")
    print(f"  Embeddings Matrix      : {embeddings_matrix.shape} ({embeddings_matrix.dtype})")
    print(f"  Archive Path           : {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"  Manifest Path          : {manifest_path}")
    print(f"  Metadata Path          : {metadata_path}")
    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute authentic NIH reference embeddings for MedTrust.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE, help="Path to training split list")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Path to dataset images folder")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH, help="Path to NIH master CSV")
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUTPUT_NPZ, help="Output .npz path")
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_PATH, help="Output manifest JSON path")
    parser.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA_PATH, help="Output metadata JSON path")
    parser.add_argument("--max-images", type=int, default=2500, help="Max images to index (default: 2500)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size (default: 64)")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic cohort selection")
    args = parser.parse_args()

    generate_reference_embeddings(
        split_file=args.split_file,
        images_dir=args.images_dir,
        csv_path=args.csv_path,
        output_path=args.out_file,
        manifest_path=args.manifest_file,
        metadata_path=args.metadata_file,
        max_images=args.max_images,
        batch_size=args.batch_size,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
