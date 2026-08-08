"""
Generate reliability diagrams for MedTrust calibration.

This script does NOT fit or modify calibration parameters.

It loads:
    - validation_predictions.csv
    - test_predictions.csv
    - frozen temperature.json

It generates:
    - 14 validation reliability diagrams
    - 14 test reliability diagrams
    - validation reliability-bin statistics
    - test reliability-bin statistics

Each diagram compares:
    1. Uncalibrated probabilities
    2. Temperature-scaled probabilities
    3. Perfect calibration reference line
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

VALIDATION_PREDICTIONS_PATH = Path(
    "outputs/evaluation/validation_predictions.csv"
)

TEST_PREDICTIONS_PATH = Path(
    "outputs/evaluation/test_predictions.csv"
)

TEMPERATURE_PATH = Path(
    "outputs/calibration/temperature.json"
)

OUTPUT_DIR = Path(
    "outputs/calibration/reliability"
)

VALIDATION_BINS_PATH = (
    OUTPUT_DIR / "validation_reliability_bins.csv"
)

TEST_BINS_PATH = (
    OUTPUT_DIR / "test_reliability_bins.csv"
)

N_BINS = 15


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
# STABLE SIGMOID
# =========================================================

def stable_sigmoid(x):

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    result = np.empty_like(
        x,
        dtype=np.float64,
    )

    positive = x >= 0
    negative = ~positive

    result[positive] = (
        1.0
        / (
            1.0
            + np.exp(-x[positive])
        )
    )

    exp_x = np.exp(
        x[negative]
    )

    result[negative] = (
        exp_x
        / (1.0 + exp_x)
    )

    return result


# =========================================================
# LOAD TEMPERATURES
# =========================================================

def load_temperatures():

    if not TEMPERATURE_PATH.exists():
        raise FileNotFoundError(
            f"Temperature artifact not found: "
            f"{TEMPERATURE_PATH}"
        )

    with open(
        TEMPERATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        artifact = json.load(f)

    if artifact.get(
        "test_data_used_for_fitting"
    ) is not False:
        raise RuntimeError(
            "Frozen calibration artifact does not confirm "
            "validation-only fitting."
        )

    if artifact.get("classes") != CLASSES:
        raise RuntimeError(
            "Temperature artifact class order mismatch."
        )

    temperatures = np.asarray(
        [
            float(
                artifact["temperatures"][
                    disease
                ]
            )
            for disease in CLASSES
        ],
        dtype=np.float64,
    )

    if not np.isfinite(
        temperatures
    ).all():
        raise RuntimeError(
            "Non-finite temperature detected."
        )

    if np.any(
        temperatures <= 0.0
    ):
        raise RuntimeError(
            "All temperatures must be positive."
        )

    return temperatures


# =========================================================
# LOAD ONE PREDICTION SPLIT
# =========================================================

def load_predictions(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction artifact not found: {path}"
        )

    targets = []
    logits = []
    probabilities = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(
                f"No CSV header in {path}."
            )

        required_columns = {
            "filename"
        }

        for disease in CLASSES:

            csv_name = CLASS_TO_CSV_NAME[
                disease
            ]

            required_columns.update({
                f"target_{csv_name}",
                f"logit_{csv_name}",
                f"prob_{csv_name}",
            })

        missing = (
            required_columns
            - set(reader.fieldnames)
        )

        if missing:
            raise ValueError(
                f"{path} missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            target_row = []
            logit_row = []
            probability_row = []

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

                probability_row.append(
                    float(
                        row[
                            f"prob_{csv_name}"
                        ]
                    )
                )

            targets.append(
                target_row
            )

            logits.append(
                logit_row
            )

            probabilities.append(
                probability_row
            )

    targets = np.asarray(
        targets,
        dtype=np.float64,
    )

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    expected_shape = (
        targets.shape[0],
        len(CLASSES),
    )

    if targets.shape != expected_shape:
        raise ValueError(
            f"Unexpected target shape: "
            f"{targets.shape}"
        )

    if logits.shape != expected_shape:
        raise ValueError(
            f"Unexpected logit shape: "
            f"{logits.shape}"
        )

    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Unexpected probability shape: "
            f"{probabilities.shape}"
        )

    if targets.shape[0] == 0:
        raise ValueError(
            f"Empty prediction artifact: {path}"
        )

    if not np.isfinite(
        targets
    ).all():
        raise ValueError(
            f"Non-finite targets in {path}."
        )

    if not np.isfinite(
        logits
    ).all():
        raise ValueError(
            f"Non-finite logits in {path}."
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            f"Non-finite probabilities in {path}."
        )

    if not np.all(
        np.isin(
            targets,
            [0.0, 1.0],
        )
    ):
        raise ValueError(
            f"Non-binary targets in {path}."
        )

    # Verify probabilities correspond to logits.
    reconstructed = stable_sigmoid(
        logits
    )

    max_difference = float(
        np.max(
            np.abs(
                reconstructed
                - probabilities
            )
        )
    )

    if max_difference > 1e-6:
        raise RuntimeError(
            "Saved probability/logit mismatch in "
            f"{path}. Max difference: "
            f"{max_difference}"
        )

    return (
        targets,
        logits,
        probabilities,
    )


# =========================================================
# CALCULATE RELIABILITY BINS
# =========================================================

def calculate_reliability_bins(
    targets,
    probabilities,
    n_bins=N_BINS,
):
    """
    Calculate equal-width reliability-bin statistics.

    Uses exactly the same binning convention as metrics.py.
    """

    targets = np.asarray(
        targets,
        dtype=np.float64,
    ).reshape(-1)

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if targets.shape != probabilities.shape:
        raise ValueError(
            "Target/probability shape mismatch."
        )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_indices = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=False,
    )

    results = []

    for bin_idx in range(n_bins):

        mask = (
            bin_indices == bin_idx
        )

        count = int(
            np.sum(mask)
        )

        lower = float(
            bin_edges[bin_idx]
        )

        upper = float(
            bin_edges[bin_idx + 1]
        )

        if count == 0:

            mean_probability = np.nan
            observed_frequency = np.nan

        else:

            mean_probability = float(
                np.mean(
                    probabilities[mask]
                )
            )

            observed_frequency = float(
                np.mean(
                    targets[mask]
                )
            )

        results.append({
            "bin_index": bin_idx,
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "mean_probability": (
                mean_probability
            ),
            "observed_frequency": (
                observed_frequency
            ),
        })

    return results


# =========================================================
# SAVE BIN STATISTICS
# =========================================================

def save_bin_statistics(
    path,
    rows,
):

    fieldnames = [
        "split",
        "disease",
        "state",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count",
        "mean_probability",
        "observed_frequency",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# PLOT ONE DISEASE
# =========================================================

def plot_reliability_diagram(
    split_name,
    disease,
    before_bins,
    after_bins,
    temperature,
):

    before_x = []
    before_y = []

    after_x = []
    after_y = []

    for row in before_bins:

        if (
            np.isfinite(
                row["mean_probability"]
            )
            and np.isfinite(
                row["observed_frequency"]
            )
        ):
            before_x.append(
                row["mean_probability"]
            )

            before_y.append(
                row["observed_frequency"]
            )

    for row in after_bins:

        if (
            np.isfinite(
                row["mean_probability"]
            )
            and np.isfinite(
                row["observed_frequency"]
            )
        ):
            after_x.append(
                row["mean_probability"]
            )

            after_y.append(
                row["observed_frequency"]
            )

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )

    # Perfect calibration reference.
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        label="Perfect calibration",
    )

    ax.plot(
        before_x,
        before_y,
        marker="o",
        linewidth=1.8,
        label="Before calibration",
    )

    ax.plot(
        after_x,
        after_y,
        marker="s",
        linewidth=1.8,
        label="After calibration",
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.set_xlabel(
        "Mean predicted probability"
    )

    ax.set_ylabel(
        "Observed positive frequency"
    )

    ax.set_title(
        f"MedTrust Reliability Diagram\n"
        f"{split_name.title()} — {disease} "
        f"(T={temperature:.3f})"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    fig.tight_layout()

    filename = (
        f"{split_name}_"
        f"{CLASS_TO_CSV_NAME[disease]}"
        ".png"
    )

    output_path = (
        OUTPUT_DIR / filename
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path


# =========================================================
# PROCESS ONE SPLIT
# =========================================================

def process_split(
    split_name,
    prediction_path,
    temperatures,
):

    (
        targets,
        logits,
        probabilities,
    ) = load_predictions(
        prediction_path
    )

    calibrated_logits = (
        logits
        / temperatures.reshape(
            1,
            -1,
        )
    )

    calibrated_probabilities = (
        stable_sigmoid(
            calibrated_logits
        )
    )

    csv_rows = []

    generated_plots = []

    for class_idx, disease in enumerate(
        CLASSES
    ):

        before_bins = (
            calculate_reliability_bins(
                targets[
                    :, class_idx
                ],
                probabilities[
                    :, class_idx
                ],
            )
        )

        after_bins = (
            calculate_reliability_bins(
                targets[
                    :, class_idx
                ],
                calibrated_probabilities[
                    :, class_idx
                ],
            )
        )

        for state, bin_rows in [
            (
                "before",
                before_bins,
            ),
            (
                "after",
                after_bins,
            ),
        ]:

            for row in bin_rows:

                csv_rows.append({
                    "split": split_name,
                    "disease": disease,
                    "state": state,
                    **row,
                })

        output_path = (
            plot_reliability_diagram(
                split_name=split_name,
                disease=disease,
                before_bins=before_bins,
                after_bins=after_bins,
                temperature=float(
                    temperatures[
                        class_idx
                    ]
                ),
            )
        )

        generated_plots.append(
            output_path
        )

    return (
        csv_rows,
        generated_plots,
        targets.shape[0],
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 78)
    print(
        "MEDTRUST — RELIABILITY DIAGRAM GENERATION"
    )
    print("=" * 78)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temperatures = (
        load_temperatures()
    )

    print(
        "ECE / reliability bins :",
        N_BINS,
    )

    print(
        "Temperatures            :",
        "FROZEN",
    )

    print(
        "Calibration fitting      :",
        "DISABLED",
    )

    print()

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print(
        "Processing validation..."
    )

    (
        validation_rows,
        validation_plots,
        validation_samples,
    ) = process_split(
        split_name="validation",
        prediction_path=(
            VALIDATION_PREDICTIONS_PATH
        ),
        temperatures=temperatures,
    )

    save_bin_statistics(
        VALIDATION_BINS_PATH,
        validation_rows,
    )

    print(
        "Validation samples :",
        validation_samples,
    )

    print(
        "Validation plots   :",
        len(validation_plots),
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    print()
    print(
        "Processing test..."
    )

    (
        test_rows,
        test_plots,
        test_samples,
    ) = process_split(
        split_name="test",
        prediction_path=(
            TEST_PREDICTIONS_PATH
        ),
        temperatures=temperatures,
    )

    save_bin_statistics(
        TEST_BINS_PATH,
        test_rows,
    )

    print(
        "Test samples       :",
        test_samples,
    )

    print(
        "Test plots         :",
        len(test_plots),
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    expected_rows = (
        len(CLASSES)
        * 2
        * N_BINS
    )

    if len(validation_rows) != expected_rows:
        raise RuntimeError(
            "Unexpected validation reliability "
            "row count."
        )

    if len(test_rows) != expected_rows:
        raise RuntimeError(
            "Unexpected test reliability "
            "row count."
        )

    if len(validation_plots) != len(CLASSES):
        raise RuntimeError(
            "Unexpected validation plot count."
        )

    if len(test_plots) != len(CLASSES):
        raise RuntimeError(
            "Unexpected test plot count."
        )

    print()
    print("=" * 78)
    print("RELIABILITY ARTIFACTS CREATED")
    print("=" * 78)

    print(
        "Validation bin data :",
        VALIDATION_BINS_PATH,
    )

    print(
        "Test bin data       :",
        TEST_BINS_PATH,
    )

    print(
        "Validation plots    :",
        len(validation_plots),
    )

    print(
        "Test plots          :",
        len(test_plots),
    )

    print(
        "Total plots         :",
        (
            len(validation_plots)
            + len(test_plots)
        ),
    )

    print()

    print(
        "Calibration parameters modified:",
        False,
    )

    print("=" * 78)


if __name__ == "__main__":
    main()