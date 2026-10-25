"""
Feature and logit extraction for MedTrust OOD detection.

This module extracts:

    - DenseNet feature vectors: 1024 dimensions
    - 14 disease logits
    - 14 sigmoid probabilities

The classifier remains frozen.

No model parameters are changed.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (
    ensure_dir,
    get_device,
    set_seed,
)
from .config import (
    FEATURE_OUTPUT_DIR,
    NUM_CLASSES,
    FEATURE_DIM,
)
from .data import (
    create_id_dataset,
    create_ood_val_dataset,
    create_ood_test_dataset,
)
from .model_adapter import MedTrustModelAdapter


# ============================================================
# DATALOADER
# ============================================================

BATCH_SIZE = 32
NUM_WORKERS = 0


def create_loader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# EXTRACT ID FEATURES
# ============================================================

def extract_id_split(
    model,
    split,
    output_path,
    batch_size=BATCH_SIZE,
):
    """
    Extract features/logits for an ID split.

    Returns:
        saved NumPy artifact path
    """

    dataset = create_id_dataset(
        split
    )

    loader = create_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_features = []
    all_logits = []
    all_probabilities = []
    all_targets = []
    all_filenames = []

    print()
    print(
        f"Extracting ID {split} features..."
    )

    for batch_idx, (
        images,
        targets,
        filenames,
    ) in enumerate(loader):

        result = model.predict(
            images
        )

        features = result[
            "features"
        ]

        logits = result[
            "logits"
        ]

        probabilities = result[
            "probabilities"
        ]

        all_features.append(
            features.cpu().numpy()
        )

        all_logits.append(
            logits.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_targets.append(
            targets.numpy()
        )

        all_filenames.extend(
            list(filenames)
        )

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

    features = np.concatenate(
        all_features,
        axis=0,
    )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    filenames = np.asarray(
        all_filenames,
        dtype=str,
    )

    validate_extracted_arrays(
        features=features,
        logits=logits,
        probabilities=probabilities,
        expected_samples=len(dataset),
    )

    output_path = Path(
        output_path
    )

    ensure_dir(
        output_path.parent
    )

    np.savez_compressed(
        output_path,
        features=features,
        logits=logits,
        probabilities=probabilities,
        targets=targets,
        filenames=filenames,
        split=split,
        source="ID",
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Features: {features.shape}"
    )

    return output_path


# ============================================================
# EXTRACT OOD FEATURES
# ============================================================

def extract_ood_split(
    model,
    split,
    output_path,
    batch_size=BATCH_SIZE,
):
    """
    Extract features/logits/probabilities for an OOD folder.
    """

    if split == "val":

        dataset = (
            create_ood_val_dataset()
        )

    elif split == "test":

        dataset = (
            create_ood_test_dataset()
        )

    else:

        raise ValueError(
            "OOD split must be 'val' or 'test'."
        )

    loader = create_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    all_features = []
    all_logits = []
    all_probabilities = []
    all_filenames = []

    print()
    print(
        f"Extracting OOD {split} features..."
    )

    for batch_idx, (
        images,
        filenames,
    ) in enumerate(loader):

        result = model.predict(
            images
        )

        features = result[
            "features"
        ]

        logits = result[
            "logits"
        ]

        probabilities = result[
            "probabilities"
        ]

        all_features.append(
            features.cpu().numpy()
        )

        all_logits.append(
            logits.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

        all_filenames.extend(
            list(filenames)
        )

        if (
            batch_idx + 1
        ) % 100 == 0:

            print(
                f"  Batch "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )

    features = np.concatenate(
        all_features,
        axis=0,
    )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    filenames = np.asarray(
        all_filenames,
        dtype=str,
    )

    validate_extracted_arrays(
        features=features,
        logits=logits,
        probabilities=probabilities,
        expected_samples=len(dataset),
    )

    output_path = Path(
        output_path
    )

    ensure_dir(
        output_path.parent
    )

    np.savez_compressed(
        output_path,
        features=features,
        logits=logits,
        probabilities=probabilities,
        filenames=filenames,
        split=split,
        source="OOD",
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Features: {features.shape}"
    )

    return output_path


# ============================================================
# VALIDATION
# ============================================================

def validate_extracted_arrays(
    features,
    logits,
    probabilities,
    expected_samples,
):

    if features.ndim != 2:

        raise ValueError(
            "Features must be 2-dimensional."
        )

    if features.shape[1] != FEATURE_DIM:

        raise ValueError(
            "Unexpected feature dimension: "
            f"{features.shape[1]} "
            f"(expected {FEATURE_DIM})."
        )

    if logits.ndim != 2:

        raise ValueError(
            "Logits must be 2-dimensional."
        )

    if logits.shape[1] != NUM_CLASSES:

        raise ValueError(
            "Unexpected number of logits: "
            f"{logits.shape[1]}"
        )

    if probabilities.shape != logits.shape:

        raise ValueError(
            "Probability/logit shape mismatch."
        )

    if features.shape[0] != expected_samples:

        raise ValueError(
            "Feature sample count mismatch."
        )

    if not np.isfinite(
        features
    ).all():

        raise ValueError(
            "Features contain NaN or Inf."
        )

    if not np.isfinite(
        logits
    ).all():

        raise ValueError(
            "Logits contain NaN or Inf."
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Probabilities contain NaN or Inf."
        )


# ============================================================
# EXTRACT EVERYTHING
# ============================================================

def extract_all():

    set_seed(42)

    device = get_device()

    print("=" * 70)
    print("MEDTRUST — OOD FEATURE EXTRACTION")
    print("=" * 70)

    print(
        "Device:",
        device,
    )

    model = MedTrustModelAdapter(
        device=device
    )

    print(
        "Model loaded."
    )

    extract_id_split(
        model=model,
        split="train",
        output_path=(
            FEATURE_OUTPUT_DIR
            / "id_train.npz"
        ),
    )

    extract_id_split(
        model=model,
        split="val",
        output_path=(
            FEATURE_OUTPUT_DIR
            / "id_val.npz"
        ),
    )

    extract_id_split(
        model=model,
        split="test",
        output_path=(
            FEATURE_OUTPUT_DIR
            / "id_test.npz"
        ),
    )

    extract_ood_split(
        model=model,
        split="val",
        output_path=(
            FEATURE_OUTPUT_DIR
            / "ood_val.npz"
        ),
    )

    extract_ood_split(
        model=model,
        split="test",
        output_path=(
            FEATURE_OUTPUT_DIR
            / "ood_test.npz"
        ),
    )

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 70)


def main():

    extract_all()


if __name__ == "__main__":

    main()