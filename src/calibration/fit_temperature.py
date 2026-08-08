"""
Fit per-disease temperature scaling for MedTrust.

Temperatures are learned using VALIDATION DATA ONLY.

Input:
    outputs/evaluation/validation_predictions.csv

Output:
    outputs/calibration/temperature.json

The final test set is never accessed by this module.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# =========================================================
# CONFIGURATION
# =========================================================

VALIDATION_PREDICTIONS_PATH = Path(
    "outputs/evaluation/validation_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/calibration"
)

TEMPERATURE_PATH = OUTPUT_DIR / "temperature.json"


CLASSES = [
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
]

# =========================================================
# CSV COLUMN NAMES
# =========================================================

CLASS_TO_CSV_NAME = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Infiltration": "infiltration",
    "Mass": "mass",
    "Nodule": "nodule",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Emphysema": "emphysema",
    "Fibrosis": "fibrosis",
    "Pleural_Thickening": "pleural_thickening",
    "Hernia": "hernia",
}

# =========================================================
# LOAD VALIDATION LOGITS AND TARGETS
# =========================================================

def load_validation_predictions():
    """
    Load validation targets and raw logits from the frozen
    validation prediction artifact.

    Returns
    -------
    targets : np.ndarray
        Shape (N, C).

    logits : np.ndarray
        Shape (N, C).

    num_samples : int
        Number of validation samples.
    """

    if not VALIDATION_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Validation prediction artifact not found: "
            f"{VALIDATION_PREDICTIONS_PATH}"
        )

    targets = []
    logits = []

    with open(
        VALIDATION_PREDICTIONS_PATH,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = set()

        for disease in CLASSES:

            csv_name = CLASS_TO_CSV_NAME[
                disease
            ]

            required_columns.add(
                f"target_{csv_name}"
            )

            required_columns.add(
                f"logit_{csv_name}"
            )

        if reader.fieldnames is None:
            raise ValueError(
                "Validation prediction CSV has no header."
            )

        missing_columns = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                "Validation prediction CSV is missing "
                f"columns: {sorted(missing_columns)}"
            )

        for row in reader:

            target_row = []
            logit_row = []

            for disease in CLASSES:

                csv_name = CLASS_TO_CSV_NAME[
                    disease
                ]

                target_row.append(
                    float(
                        row[
                            f"target_{csv_name}"
                        ]
                    )
                )

                logit_row.append(
                    float(
                        row[
                            f"logit_{csv_name}"
                        ]
                    )
                )

            targets.append(target_row)
            logits.append(logit_row)

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    verify_loaded_data(
        targets=targets,
        logits=logits,
    )

    return targets, logits, len(targets)


# =========================================================
# DATA VERIFICATION
# =========================================================

def verify_loaded_data(
    targets,
    logits,
):
    """
    Verify the loaded validation calibration data.
    """

    if targets.ndim != 2:
        raise ValueError(
            f"Targets must be 2-D, got {targets.shape}."
        )

    if logits.ndim != 2:
        raise ValueError(
            f"Logits must be 2-D, got {logits.shape}."
        )

    if targets.shape != logits.shape:
        raise ValueError(
            "Target/logit shape mismatch: "
            f"{targets.shape} vs {logits.shape}."
        )

    if targets.shape[1] != len(CLASSES):
        raise ValueError(
            "Unexpected number of disease columns: "
            f"{targets.shape[1]}"
        )

    if targets.shape[0] == 0:
        raise ValueError(
            "Validation calibration data is empty."
        )

    if not np.isfinite(targets).all():
        raise ValueError(
            "Non-finite validation targets detected."
        )

    if not np.isfinite(logits).all():
        raise ValueError(
            "Non-finite validation logits detected."
        )

    if not np.all(
        np.isin(targets, [0.0, 1.0])
    ):
        raise ValueError(
            "Validation targets must be binary."
        )

    for class_idx, disease in enumerate(CLASSES):

        unique_targets = np.unique(
            targets[:, class_idx]
        )

        if len(unique_targets) < 2:
            raise ValueError(
                f"{disease} does not contain both "
                "positive and negative validation samples."
            )


# =========================================================
# FIT ONE TEMPERATURE
# =========================================================

def fit_temperature_for_class(
    logits,
    targets,
):
    """
    Fit one positive scalar temperature by minimizing
    binary negative log-likelihood.

    Optimization is performed over log(T), ensuring:

        T = exp(log(T)) > 0
    """

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float64,
    )

    targets_tensor = torch.tensor(
        targets,
        dtype=torch.float64,
    )

    log_temperature = torch.zeros(
        1,
        dtype=torch.float64,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(
            log_temperature
        )

        calibrated_logits = (
            logits_tensor / temperature
        )

        loss = F.binary_cross_entropy_with_logits(
            calibrated_logits,
            targets_tensor,
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    with torch.no_grad():

        temperature = float(
            torch.exp(
                log_temperature
            ).item()
        )

        before_nll = float(
            F.binary_cross_entropy_with_logits(
                logits_tensor,
                targets_tensor,
            ).item()
        )

        calibrated_logits = (
            logits_tensor / temperature
        )

        after_nll = float(
            F.binary_cross_entropy_with_logits(
                calibrated_logits,
                targets_tensor,
            ).item()
        )

    if not np.isfinite(temperature):
        raise RuntimeError(
            "Temperature optimization produced "
            "a non-finite value."
        )

    if temperature <= 0.0:
        raise RuntimeError(
            "Temperature optimization produced "
            "a non-positive value."
        )

    # Temperature scaling should never make the optimized
    # validation NLL meaningfully worse than T = 1.
    if after_nll > before_nll + 1e-10:
        raise RuntimeError(
            "Temperature optimization increased "
            "validation NLL."
        )

    return {
        "temperature": temperature,
        "nll_before": before_nll,
        "nll_after": after_nll,
        "nll_improvement": (
            before_nll - after_nll
        ),
    }


# =========================================================
# FIT ALL DISEASE TEMPERATURES
# =========================================================

def fit_all_temperatures(
    targets,
    logits,
):
    """
    Fit one independent scalar temperature for each disease.
    """

    results = {}

    for class_idx, disease in enumerate(
        CLASSES
    ):

        result = fit_temperature_for_class(
            logits=logits[:, class_idx],
            targets=targets[:, class_idx],
        )

        results[disease] = result

    return results


# =========================================================
# SAVE FROZEN CALIBRATION ARTIFACT
# =========================================================

def save_temperature_artifact(
    results,
    num_samples,
):
    """
    Save the fitted validation-only temperatures.

    Existing calibration artifacts are never overwritten
    automatically.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TEMPERATURE_PATH.exists():
        raise FileExistsError(
            "Temperature artifact already exists: "
            f"{TEMPERATURE_PATH}\n"
            "Refusing to overwrite frozen calibration."
        )

    temperature_map = {
        disease: float(
            results[disease]["temperature"]
        )
        for disease in CLASSES
    }

    optimization_details = {
        disease: {
            "nll_before": float(
                results[disease]["nll_before"]
            ),
            "nll_after": float(
                results[disease]["nll_after"]
            ),
            "nll_improvement": float(
                results[disease][
                    "nll_improvement"
                ]
            ),
        }
        for disease in CLASSES
    }

    artifact = {
        "project": "MedTrust",
        "module": "Probability Calibration",

        "method": "per-disease temperature scaling",

        "calibration_split": "validation",
        "calibration_source": str(
            VALIDATION_PREDICTIONS_PATH
        ),
        "calibration_samples": int(
            num_samples
        ),

        "objective": (
            "binary negative log-likelihood"
        ),

        "parameterization": (
            "temperature = exp(log_temperature)"
        ),

        "classes": list(CLASSES),

        "temperatures": temperature_map,

        "validation_optimization": (
            optimization_details
        ),

        "test_data_used_for_fitting": False,
    }

    with open(
        TEMPERATURE_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            artifact,
            f,
            indent=4,
        )

    return artifact


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 78)
    print("MEDTRUST — VALIDATION TEMPERATURE SCALING")
    print("=" * 78)

    print(
        "Source :",
        VALIDATION_PREDICTIONS_PATH,
    )

    print(
        "Output :",
        TEMPERATURE_PATH,
    )

    print()

    targets, logits, num_samples = (
        load_validation_predictions()
    )

    print("Validation calibration data")
    print("-" * 78)

    print(
        "Samples :",
        num_samples,
    )

    print(
        "Targets :",
        targets.shape,
    )

    print(
        "Logits  :",
        logits.shape,
    )

    print()

    print(
        "Fitting one temperature per disease..."
    )

    print()

    results = fit_all_temperatures(
        targets=targets,
        logits=logits,
    )

    print("=" * 78)
    print("TEMPERATURE SCALING RESULTS")
    print("=" * 78)

    print(
        f"{'Disease':20}"
        f"{'T':>12}"
        f"{'NLL Before':>15}"
        f"{'NLL After':>15}"
        f"{'Improvement':>15}"
    )

    print("-" * 78)

    for disease in CLASSES:

        result = results[disease]

        print(
            f"{disease:20}"
            f"{result['temperature']:12.6f}"
            f"{result['nll_before']:15.8f}"
            f"{result['nll_after']:15.8f}"
            f"{result['nll_improvement']:15.8f}"
        )

    print("-" * 78)

    artifact = save_temperature_artifact(
        results=results,
        num_samples=num_samples,
    )

    print()

    print(
        "Frozen calibration artifact saved:"
    )

    print(
        TEMPERATURE_PATH
    )

    print()

    print(
        "Test data used for fitting:",
        artifact[
            "test_data_used_for_fitting"
        ],
    )

    print("=" * 78)


if __name__ == "__main__":
    main()