"""
Configuration for the MedTrust OOD module.

The OOD module is intentionally post-hoc:
the trained DenseNet-121 classifier is frozen and is not retrained.
"""

from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# EXISTING MEDTRUST ARTIFACTS
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_model.pt"
)

DATASET_DIR = (
    PROJECT_ROOT
    / "dataset"
)

IMAGE_DIR = (
    DATASET_DIR
    / "images"
)

METADATA_CSV = (
    DATASET_DIR
    / "Data_Entry_2017_v2020.csv"
)

TRAIN_SPLIT = (
    PROJECT_ROOT
    / "splits"
    / "train_list.txt"
)

VAL_SPLIT = (
    PROJECT_ROOT
    / "splits"
    / "val_list.txt"
)

TEST_SPLIT = (
    PROJECT_ROOT
    / "splits"
    / "test_list.txt"
)


# ============================================================
# OOD DATA
# ============================================================

OOD_DATA_DIR = (
    PROJECT_ROOT
    / "ood_data"
)

OOD_VAL_DIR = (
    OOD_DATA_DIR
    / "val"
)

OOD_TEST_DIR = (
    OOD_DATA_DIR
    / "test"
)


# ============================================================
# OOD OUTPUTS
# ============================================================

OOD_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "ood"
)

FEATURE_OUTPUT_DIR = (
    OOD_OUTPUT_DIR
    / "features"
)

SCORE_OUTPUT_DIR = (
    OOD_OUTPUT_DIR
    / "scores"
)

EVALUATION_OUTPUT_DIR = (
    OOD_OUTPUT_DIR
    / "evaluation"
)

PLOT_OUTPUT_DIR = (
    OOD_OUTPUT_DIR
    / "plots"
)

ARTIFACT_OUTPUT_DIR = (
    OOD_OUTPUT_DIR
    / "artifacts"
)


# ============================================================
# MODEL
# ============================================================

NUM_CLASSES = 14

FEATURE_DIM = 1024


# ============================================================
# OOD SETTINGS
# ============================================================

# Temperature used by the energy score.
ENERGY_TEMPERATURE = 1.0

# Numerical stability for Mahalanobis variance.
MAHALANOBIS_EPSILON = 1e-6


# ============================================================
# RANDOMNESS
# ============================================================

RANDOM_SEED = 42


# ============================================================
# THRESHOLDING
# ============================================================

# Higher OOD score means "more OOD".
THRESHOLD_METHOD = "youden_j"



THRESHOLD_OUTPUT_PATH = (
    ARTIFACT_OUTPUT_DIR
    / "ood_thresholds.json"
)



# ============================================================
# DEVICE
# ============================================================

USE_CUDA_IF_AVAILABLE = True