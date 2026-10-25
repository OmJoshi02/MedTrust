
"""Calibrate OOD thresholds using NIH ID validation data only."""

import numpy as np

from .common import ensure_dir
from .config import (
    FEATURE_OUTPUT_DIR,
    ARTIFACT_OUTPUT_DIR,
    MAHALANOBIS_EPSILON,
)
from .scores import (
    msp_score,
    energy_score,
    mahalanobis_score,
)
from .threshold import (
    select_threshold_id_percentile,
    save_thresholds,
)


TARGET_FALSE_POSITIVE_RATE = 0.10

TRAIN_PATH = FEATURE_OUTPUT_DIR / "id_train.npz"
VAL_PATH = FEATURE_OUTPUT_DIR / "id_val.npz"

MAHALANOBIS_PATH = (
    ARTIFACT_OUTPUT_DIR / "mahalanobis_statistics.npz"
)


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            "Run: python -m src.OOD.extract_id_calibration"
        )

    with np.load(path, allow_pickle=False) as data:
        return {
            key: np.asarray(data[key])
            for key in data.files
        }


def require_finite(name, array):
    if array.size == 0:
        raise ValueError(f"{name} is empty.")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")


def main():
    print("=" * 70)
    print("MEDTRUST OOD — ID-ONLY PERCENTILE CALIBRATION")
    print("=" * 70)

    train = load_npz(TRAIN_PATH)
    val = load_npz(VAL_PATH)

    for key in ("features",):
        if key not in train:
            raise KeyError(f"Training artifact missing '{key}'.")

    for key in ("features", "logits"):
        if key not in val:
            raise KeyError(f"Validation artifact missing '{key}'.")

    train_features = np.asarray(
        train["features"], dtype=np.float64
    )
    val_features = np.asarray(
        val["features"], dtype=np.float64
    )
    val_logits = np.asarray(
        val["logits"], dtype=np.float64
    )

    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("Feature arrays must be 2-dimensional.")

    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("Training/validation feature dimensions differ.")

    if val_features.shape[0] != val_logits.shape[0]:
        raise ValueError("Validation feature/logit sample counts differ.")

    require_finite("Training features", train_features)
    require_finite("Validation features", val_features)
    require_finite("Validation logits", val_logits)

    # Fit diagonal Mahalanobis statistics using ID TRAIN only.
    mean = np.mean(train_features, axis=0)
    variance = np.var(train_features, axis=0)
    safe_variance = variance + MAHALANOBIS_EPSILON

    if np.any(safe_variance <= 0):
        raise ValueError("Mahalanobis safe variance must be positive.")

    ensure_dir(ARTIFACT_OUTPUT_DIR)

    np.savez_compressed(
        MAHALANOBIS_PATH,
        mean=mean,
        variance=variance,
        safe_variance=safe_variance,
    )

    print("Saved Mahalanobis statistics:", MAHALANOBIS_PATH)
    print("ID training samples:", len(train_features))
    print("ID validation samples:", len(val_logits))

    # All three methods use the same frozen model outputs.
    validation_scores = {
        "msp": msp_score(val_logits),
        "energy": energy_score(val_logits),
        "mahalanobis": mahalanobis_score(
            features=val_features,
            mean=mean,
            safe_variance=safe_variance,
        ),
    }

    thresholds = {}

    for method, scores in validation_scores.items():
        result = select_threshold_id_percentile(
            id_scores=scores,
            target_false_positive_rate=(
                TARGET_FALSE_POSITIVE_RATE
            ),
        )

        thresholds[method] = result

        print()
        print(f"Method: {method}")
        print(f"  Threshold: {result['threshold']:.8f}")
        print(
            "  Target false-positive rate: "
            f"{result['target_false_positive_rate']:.1%}"
        )
        print(
            "  Observed validation false-positive rate: "
            f"{result['observed_false_positive_rate']:.1%}"
        )

    save_thresholds(
        thresholds=thresholds,
        threshold_method="id_percentile",
    )

    print()
    print("ID-only threshold calibration complete.")
    print("OOD validation images used: 0")
    print("Test images used: 0")
    print("=" * 70)


if __name__ == "__main__":
    main()
