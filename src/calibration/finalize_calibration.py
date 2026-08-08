"""
Finalize and audit the MedTrust calibration stage.

This script does NOT:
    - fit temperatures
    - modify temperatures
    - tune using the test set
    - recalculate predictions

It verifies existing calibration artifacts and creates one
authoritative final calibration summary.
"""

import csv
import json
from pathlib import Path


# =========================================================
# PATHS
# =========================================================

CALIBRATION_DIR = Path(
    "outputs/calibration"
)

TEMPERATURE_PATH = (
    CALIBRATION_DIR / "temperature.json"
)

SUMMARY_PATH = (
    CALIBRATION_DIR / "calibration_summary.json"
)

VALIDATION_METRICS_PATH = (
    CALIBRATION_DIR
    / "validation_calibration_metrics.csv"
)

TEST_METRICS_PATH = (
    CALIBRATION_DIR
    / "test_calibration_metrics.csv"
)

ECE_SENSITIVITY_PATH = (
    CALIBRATION_DIR
    / "ece_sensitivity.csv"
)

RELIABILITY_DIR = (
    CALIBRATION_DIR / "reliability"
)

VALIDATION_RELIABILITY_BINS = (
    RELIABILITY_DIR
    / "validation_reliability_bins.csv"
)

TEST_RELIABILITY_BINS = (
    RELIABILITY_DIR
    / "test_reliability_bins.csv"
)

FINAL_SUMMARY_PATH = (
    CALIBRATION_DIR
    / "calibration_final_summary.json"
)


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
# BASIC HELPERS
# =========================================================

def require_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact missing: {path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Expected file but found something else: {path}"
        )


def load_json(path):

    require_file(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def load_csv(path):

    require_file(path)

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        return list(
            csv.DictReader(f)
        )


# =========================================================
# VERIFY TEMPERATURE ARTIFACT
# =========================================================

def verify_temperature_artifact():

    artifact = load_json(
        TEMPERATURE_PATH
    )

    if artifact.get(
        "calibration_split"
    ) != "validation":

        raise RuntimeError(
            "Temperatures were not marked as "
            "validation-fitted."
        )

    if artifact.get(
        "test_data_used_for_fitting"
    ) is not False:

        raise RuntimeError(
            "Temperature artifact does not confirm "
            "that test data was excluded from fitting."
        )

    if artifact.get(
        "classes"
    ) != CLASSES:

        raise RuntimeError(
            "Temperature artifact class order mismatch."
        )

    temperatures = artifact.get(
        "temperatures"
    )

    if not isinstance(
        temperatures,
        dict,
    ):

        raise RuntimeError(
            "Temperature map missing."
        )

    if set(
        temperatures.keys()
    ) != set(CLASSES):

        raise RuntimeError(
            "Temperature disease set mismatch."
        )

    for disease in CLASSES:

        value = float(
            temperatures[disease]
        )

        if value <= 0:
            raise RuntimeError(
                f"Invalid temperature for {disease}: "
                f"{value}"
            )

    return artifact


# =========================================================
# VERIFY CALIBRATION SUMMARY
# =========================================================

def verify_calibration_summary():

    summary = load_json(
        SUMMARY_PATH
    )

    if summary.get(
        "calibration_split"
    ) != "validation":

        raise RuntimeError(
            "Calibration summary split mismatch."
        )

    if summary.get(
        "test_data_used_for_fitting"
    ) is not False:

        raise RuntimeError(
            "Calibration summary does not confirm "
            "test exclusion from fitting."
        )

    for split in [
        "validation",
        "test",
    ]:

        if split not in summary:
            raise RuntimeError(
                f"Missing {split} calibration summary."
            )

        for metric in [
            "nll",
            "brier",
            "ece",
            "auroc",
        ]:

            if metric not in summary[split]:
                raise RuntimeError(
                    f"Missing {split}/{metric}."
                )

    return summary


# =========================================================
# VERIFY PER-DISEASE METRICS
# =========================================================

def verify_metric_csv(
    path,
):

    rows = load_csv(path)

    if len(rows) != len(CLASSES):
        raise RuntimeError(
            f"{path} should contain "
            f"{len(CLASSES)} disease rows, "
            f"found {len(rows)}."
        )

    diseases = [
        row["disease"]
        for row in rows
    ]

    if diseases != CLASSES:
        raise RuntimeError(
            f"Disease order mismatch in {path}."
        )

    return rows


# =========================================================
# VERIFY ECE SENSITIVITY
# =========================================================

def verify_ece_sensitivity():

    rows = load_csv(
        ECE_SENSITIVITY_PATH
    )

    expected_rows = (
        2 * 14 * 3
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            "ECE sensitivity row count mismatch: "
            f"expected {expected_rows}, "
            f"found {len(rows)}."
        )

    expected_bins = {
        10,
        15,
        20,
    }

    for split in [
        "validation",
        "test",
    ]:

        split_rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        if len(split_rows) != 42:
            raise RuntimeError(
                f"Unexpected ECE rows for {split}."
            )

        bins = {
            int(row["n_bins"])
            for row in split_rows
        }

        if bins != expected_bins:
            raise RuntimeError(
                f"Unexpected ECE bin settings "
                f"for {split}: {bins}"
            )

    return rows


# =========================================================
# VERIFY RELIABILITY ARTIFACTS
# =========================================================

def verify_reliability_artifacts():

    validation_rows = load_csv(
        VALIDATION_RELIABILITY_BINS
    )

    test_rows = load_csv(
        TEST_RELIABILITY_BINS
    )

    expected_bin_rows = (
        14 * 2 * 15
    )

    if len(
        validation_rows
    ) != expected_bin_rows:

        raise RuntimeError(
            "Validation reliability-bin "
            "row count mismatch."
        )

    if len(
        test_rows
    ) != expected_bin_rows:

        raise RuntimeError(
            "Test reliability-bin "
            "row count mismatch."
        )

    validation_plots = list(
        RELIABILITY_DIR.glob(
            "validation_*.png"
        )
    )

    test_plots = list(
        RELIABILITY_DIR.glob(
            "test_*.png"
        )
    )

    if len(
        validation_plots
    ) != 14:

        raise RuntimeError(
            "Expected 14 validation "
            "reliability diagrams, found "
            f"{len(validation_plots)}."
        )

    if len(
        test_plots
    ) != 14:

        raise RuntimeError(
            "Expected 14 test "
            "reliability diagrams, found "
            f"{len(test_plots)}."
        )

    return {
        "validation_bin_rows": len(
            validation_rows
        ),
        "test_bin_rows": len(
            test_rows
        ),
        "validation_plots": len(
            validation_plots
        ),
        "test_plots": len(
            test_plots
        ),
        "total_plots": (
            len(validation_plots)
            + len(test_plots)
        ),
    }


# =========================================================
# SUMMARIZE ECE SENSITIVITY
# =========================================================

def summarize_ece_sensitivity(
    rows,
):

    result = {}

    for split in [
        "validation",
        "test",
    ]:

        result[split] = {}

        for n_bins in [
            10,
            15,
            20,
        ]:

            selected = [
                row
                for row in rows
                if (
                    row["split"] == split
                    and int(
                        row["n_bins"]
                    ) == n_bins
                )
            ]

            before = sum(
                float(
                    row["ece_before"]
                )
                for row in selected
            ) / len(selected)

            after = sum(
                float(
                    row["ece_after"]
                )
                for row in selected
            ) / len(selected)

            result[split][
                str(n_bins)
            ] = {
                "before": before,
                "after": after,
                "change": (
                    after - before
                ),
            }

    return result


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 78)
    print(
        "MEDTRUST — FINAL CALIBRATION AUDIT"
    )
    print("=" * 78)

    if FINAL_SUMMARY_PATH.exists():

        raise FileExistsError(
            "Final calibration summary already exists: "
            f"{FINAL_SUMMARY_PATH}\n"
            "Refusing to overwrite frozen final audit."
        )

    print(
        "Verifying temperature artifact..."
    )

    temperature_artifact = (
        verify_temperature_artifact()
    )

    print("  PASSED")

    print(
        "Verifying calibration summary..."
    )

    calibration_summary = (
        verify_calibration_summary()
    )

    print("  PASSED")

    print(
        "Verifying validation metrics..."
    )

    validation_metrics = (
        verify_metric_csv(
            VALIDATION_METRICS_PATH
        )
    )

    print(
        f"  PASSED ({len(validation_metrics)} diseases)"
    )

    print(
        "Verifying test metrics..."
    )

    test_metrics = (
        verify_metric_csv(
            TEST_METRICS_PATH
        )
    )

    print(
        f"  PASSED ({len(test_metrics)} diseases)"
    )

    print(
        "Verifying ECE sensitivity..."
    )

    ece_rows = (
        verify_ece_sensitivity()
    )

    print(
        f"  PASSED ({len(ece_rows)} rows)"
    )

    print(
        "Verifying reliability artifacts..."
    )

    reliability = (
        verify_reliability_artifacts()
    )

    print(
        "  PASSED "
        f"({reliability['total_plots']} plots)"
    )

    ece_sensitivity = (
        summarize_ece_sensitivity(
            ece_rows
        )
    )

    # -----------------------------------------------------
    # Final scientific interpretation
    # -----------------------------------------------------

    test = calibration_summary[
        "test"
    ]

    nll_improved = (
        test["nll"]["after"]
        < test["nll"]["before"]
    )

    brier_improved = (
        test["brier"]["after"]
        < test["brier"]["before"]
    )

    ece_improved = (
        test["ece"]["after"]
        < test["ece"]["before"]
    )

    auroc_change = abs(
        test["auroc"]["change"]
    )

    auroc_preserved = (
        auroc_change < 1e-6
    )

    ece_changes = [
        ece_sensitivity["test"][
            str(n_bins)
        ]["change"]
        for n_bins in [
            10,
            15,
            20,
        ]
    ]

    ece_worsened_all_bins = all(
        value > 0
        for value in ece_changes
    )

    # -----------------------------------------------------
    # Final artifact
    # -----------------------------------------------------

    final_artifact = {
        "project": "MedTrust",

        "stage": "Probability Calibration",

        "status": "COMPLETED_AND_FROZEN",

        "method": (
            "Per-disease temperature scaling"
        ),

        "number_of_diseases": len(
            CLASSES
        ),

        "calibration_fit_split": (
            "validation"
        ),

        "calibration_samples": (
            temperature_artifact[
                "calibration_samples"
            ]
        ),

        "validation_samples": (
            calibration_summary[
                "validation_samples"
            ]
        ),

        "test_samples": (
            calibration_summary[
                "test_samples"
            ]
        ),

        "test_data_used_for_fitting": False,

        "temperatures": (
            temperature_artifact[
                "temperatures"
            ]
        ),

        "macro_validation_metrics": (
            calibration_summary[
                "validation"
            ]
        ),

        "macro_test_metrics": (
            calibration_summary[
                "test"
            ]
        ),

        "ece_sensitivity": (
            ece_sensitivity
        ),

        "reliability_analysis": (
            reliability
        ),

        "test_findings": {
            "nll_improved": (
                nll_improved
            ),
            "brier_improved": (
                brier_improved
            ),
            "ece_improved_15_bins": (
                ece_improved
            ),
            "ece_worsened_for_10_15_20_bins": (
                ece_worsened_all_bins
            ),
            "auroc_preserved": (
                auroc_preserved
            ),
        },

        "interpretation": (
            "Validation-fitted per-disease temperature "
            "scaling improved test NLL and Brier score "
            "while preserving AUROC. Equal-width ECE "
            "increased, and this increase remained "
            "consistent for 10, 15, and 20 bins. "
            "Therefore calibration provides improved "
            "proper-scoring-rule performance but mixed "
            "evidence under binned ECE."
        ),

        "frozen_artifacts": {
            "temperature": str(
                TEMPERATURE_PATH
            ),
            "calibration_summary": str(
                SUMMARY_PATH
            ),
            "validation_metrics": str(
                VALIDATION_METRICS_PATH
            ),
            "test_metrics": str(
                TEST_METRICS_PATH
            ),
            "ece_sensitivity": str(
                ECE_SENSITIVITY_PATH
            ),
            "validation_reliability_bins": str(
                VALIDATION_RELIABILITY_BINS
            ),
            "test_reliability_bins": str(
                TEST_RELIABILITY_BINS
            ),
            "reliability_directory": str(
                RELIABILITY_DIR
            ),
        },

        "calibration_parameters_modified_during_audit": (
            False
        ),

        "test_set_policy": (
            "The final test set was used only for "
            "evaluation and was not used to fit or "
            "select temperature parameters."
        ),
    }

    with open(
        FINAL_SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_artifact,
            f,
            indent=4,
        )

    # -----------------------------------------------------
    # Display final result
    # -----------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL TEST CALIBRATION RESULT")
    print("=" * 78)

    print(
        "NLL   :",
        f"{test['nll']['before']:.8f}",
        "->",
        f"{test['nll']['after']:.8f}",
        "IMPROVED"
        if nll_improved
        else "NOT IMPROVED",
    )

    print(
        "Brier :",
        f"{test['brier']['before']:.8f}",
        "->",
        f"{test['brier']['after']:.8f}",
        "IMPROVED"
        if brier_improved
        else "NOT IMPROVED",
    )

    print(
        "ECE   :",
        f"{test['ece']['before']:.8f}",
        "->",
        f"{test['ece']['after']:.8f}",
        "IMPROVED"
        if ece_improved
        else "WORSENED",
    )

    print(
        "AUROC :",
        f"{test['auroc']['before']:.8f}",
        "->",
        f"{test['auroc']['after']:.8f}",
        "PRESERVED"
        if auroc_preserved
        else "REVIEW REQUIRED",
    )

    print()
    print(
        "ECE worsened for 10/15/20 bins:",
        ece_worsened_all_bins,
    )

    print(
        "Test data used for fitting:",
        False,
    )

    print()
    print("=" * 78)
    print(
        "CALIBRATION STATUS: COMPLETED AND FROZEN"
    )
    print("=" * 78)

    print(
        "Final audit artifact:"
    )

    print(
        FINAL_SUMMARY_PATH
    )

    print()
    print(
        "No calibration parameters were modified."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()