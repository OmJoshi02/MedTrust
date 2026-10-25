"""
Model and artifact verification for MedTrust OOD Detection.
"""

from pathlib import Path
import json
import numpy as np
import torch

from .config import (
    CHECKPOINT_PATH,
    ARTIFACT_OUTPUT_DIR,
    FEATURE_DIM,
    NUM_CLASSES,
)
from .common import load_json
from .threshold import THRESHOLD_OUTPUT_PATH


MAHALANOBIS_PATH = ARTIFACT_OUTPUT_DIR / "mahalanobis_statistics.npz"


def verify_checkpoint():
    """Verify that the frozen MedTrust model checkpoint exists and is valid."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"DenseNet-121 checkpoint missing at: {CHECKPOINT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    if "classifier.weight" not in state:
        raise KeyError("Classifier weights missing in checkpoint state dictionary.")

    in_feat = state["classifier.weight"].shape[1]
    out_feat = state["classifier.weight"].shape[0]

    if in_feat != FEATURE_DIM or out_feat != NUM_CLASSES:
        raise ValueError(
            f"Classifier dimension mismatch: got ({out_feat}, {in_feat}), "
            f"expected ({NUM_CLASSES}, {FEATURE_DIM})"
        )
    return True


def verify_thresholds():
    """Verify that frozen/promoted OOD threshold artifact exists and is valid."""
    if not THRESHOLD_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Threshold artifact missing at: {THRESHOLD_OUTPUT_PATH}")

    data = load_json(THRESHOLD_OUTPUT_PATH)
    thresholds = data.get("thresholds", {})

    required_methods = {"msp", "energy", "mahalanobis"}
    missing = required_methods - set(thresholds.keys())
    if missing:
        raise KeyError(f"Missing methods in threshold artifact: {missing}")

    return {m: thresholds[m]["threshold"] for m in required_methods}


def verify_mahalanobis_statistics():
    """Verify Mahalanobis reference statistics file."""
    if not MAHALANOBIS_PATH.exists():
        raise FileNotFoundError(f"Mahalanobis statistics missing at: {MAHALANOBIS_PATH}")

    with np.load(MAHALANOBIS_PATH, allow_pickle=False) as data:
        required = {"mean", "safe_variance"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(f"Missing arrays in Mahalanobis artifact: {missing}")

        mean = data["mean"]
        safe_var = data["safe_variance"]

        if mean.shape[0] != FEATURE_DIM or safe_var.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"Dimension mismatch in Mahalanobis stats: got {mean.shape[0]}, "
                f"expected {FEATURE_DIM}"
            )
    return True


def main():
    print("=" * 70)
    print("MEDTRUST OOD MODULE VERIFICATION")
    print("=" * 70)

    try:
        verify_checkpoint()
        print("[OK] DenseNet-121 Checkpoint verified (checkpoints/best_model.pt)")

        thresh_map = verify_thresholds()
        print("[OK] Threshold artifact verified (outputs/ood/artifacts/ood_thresholds.json)")
        print(f"     - MSP Threshold         : {thresh_map['msp']:.6f}")
        print(f"     - Energy Threshold      : {thresh_map['energy']:.6f}")
        print(f"     - Mahalanobis Threshold : {thresh_map['mahalanobis']:.6f}")

        verify_mahalanobis_statistics()
        print("[OK] Mahalanobis Statistics verified (outputs/ood/artifacts/mahalanobis_statistics.npz)")
        print("=" * 70)
        print("ALL OOD ARTIFACTS AND MODEL DEPENDENCIES ARE HEALTHY AND READY.")
        print("=" * 70)
        return True
    except Exception as exc:
        print(f"[FAIL] Verification failed: {exc}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    main()

