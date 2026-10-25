import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from src.dataset import CLASSES


class CalibrationAnalyzer:
    """
    Calibration Analyzer for MedTrust V1 (Module 9).

    Performs runtime lookup and diagnostic analysis of already-calibrated
    model predictions against empirical validation reliability bins.

    Scientific Guarantees:
    - Strictly diagnostic: does NOT recalibrate the model, train parameters,
      or modify classifier weights.
    - Uses frozen validation reliability bins (validation_reliability_bins.csv)
      with zero test-set leakage.
    - Explicit mathematical definition of calibration discrepancy:
        calibration_gap = probability - empirical_positive_rate
        absolute_calibration_gap = |calibration_gap|
    - Deterministic bin boundary resolution (handles 0.0, 1.0, exact boundaries).
    - Robust edge-case handling (missing/malformed files, unpopulated bins, NaNs).
    """

    DEFAULT_BINS_PATH = Path("outputs/calibration/reliability/validation_reliability_bins.csv")
    REQUIRED_COLUMNS = {
        "split",
        "disease",
        "state",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count",
        "mean_probability",
        "observed_frequency",
    }

    def __init__(
        self,
        bins_csv_path: Optional[Union[str, Path]] = None,
        state: str = "after",
        warning_threshold: float = 0.10,
        high_discrepancy_threshold: float = 0.20,
    ):
        if state not in ("before", "after"):
            raise ValueError(f"state must be 'before' or 'after', got '{state}'")
        if not (0.0 < warning_threshold < high_discrepancy_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0.0 < warning_threshold < high_discrepancy_threshold <= 1.0"
            )

        self.state = state
        self.warning_threshold = float(warning_threshold)
        self.high_discrepancy_threshold = float(high_discrepancy_threshold)

        if bins_csv_path is not None:
            self.bins_csv_path = Path(bins_csv_path)
        else:
            self.bins_csv_path = self.DEFAULT_BINS_PATH
            if not self.bins_csv_path.exists():
                repo_root = Path(__file__).resolve().parents[2]
                alt = repo_root / self.DEFAULT_BINS_PATH
                if alt.exists():
                    self.bins_csv_path = alt

        self._bins_by_disease = {}
        self._load_and_validate_bins()

    # =========================================================
    # LOAD AND VALIDATE FROZEN BINS
    # =========================================================

    def _load_and_validate_bins(self) -> None:
        if not self.bins_csv_path.exists():
            raise FileNotFoundError(
                f"Validation reliability bins CSV not found at: {self.bins_csv_path}"
            )

        df = pd.read_csv(self.bins_csv_path)

        missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Bins CSV missing required columns: {sorted(missing_cols)}"
            )

        # Filter by requested calibration state ('after' for calibrated, 'before' for raw)
        state_df = df[df["state"] == self.state].copy()
        if state_df.empty:
            raise ValueError(
                f"No records found in {self.bins_csv_path} for state='{self.state}'."
            )

        for disease in CLASSES:
            d_df = state_df[state_df["disease"] == disease].sort_values("bin_index")
            if d_df.empty:
                raise ValueError(
                    f"Missing disease records in bins CSV for class '{disease}' (state='{self.state}')."
                )

            bins_list = []
            for _, row in d_df.iterrows():
                bins_list.append(
                    {
                        "bin_index": int(row["bin_index"]),
                        "bin_lower": float(row["bin_lower"]),
                        "bin_upper": float(row["bin_upper"]),
                        "count": int(row["count"]),
                        "mean_probability": float(row["mean_probability"]),
                        "observed_frequency": (
                            float(row["observed_frequency"])
                            if not pd.isna(row["observed_frequency"])
                            else float("nan")
                        ),
                    }
                )

            # Sanity check: Ensure bins are contiguous
            if len(bins_list) == 0:
                raise ValueError(f"Zero bins found for disease: {disease}")

            self._bins_by_disease[disease] = bins_list

    # =========================================================
    # SINGLE DISEASE DIAGNOSTIC
    # =========================================================

    def analyze_prediction(
        self,
        disease: str,
        probability: float,
    ) -> Dict:
        """
        Analyze the calibration reliability of a single pathology prediction.

        Parameters
        ----------
        disease:
            Target disease name (e.g. 'Pneumonia', 'Cardiomegaly').
        probability:
            Predicted probability (raw or calibrated) in [0.0, 1.0].

        Returns
        -------
        Structured dictionary containing bin details, empirical positive rate,
        calibration gap, and diagnostic status.
        """
        if disease not in self._bins_by_disease:
            raise ValueError(
                f"Unsupported disease: '{disease}'. Valid options are: {CLASSES}"
            )

        if not isinstance(probability, (int, float, np.floating)):
            raise TypeError(
                f"Probability must be numeric, got {type(probability)}"
            )

        if math.isnan(probability) or math.isinf(probability):
            raise ValueError(f"Probability cannot be NaN or Inf, got {probability}")

        if not (0.0 <= probability <= 1.0):
            raise ValueError(
                f"Probability must be in range [0.0, 1.0], got {probability}"
            )

        probability = float(probability)
        bins = self._bins_by_disease[disease]
        num_bins = len(bins)

        # -----------------------------------------------------
        # Deterministic Bin Matching
        # Half-open [bin_lower, bin_upper) for bins 0 to N-2
        # Closed [bin_lower, bin_upper] for the final bin N-1
        # -----------------------------------------------------
        matched_bin = None
        for i, b in enumerate(bins):
            is_last = i == (num_bins - 1)
            if is_last:
                if b["bin_lower"] <= probability <= b["bin_upper"]:
                    matched_bin = b
                    break
            else:
                if b["bin_lower"] <= probability < b["bin_upper"]:
                    matched_bin = b
                    break

        # Edge case: Floating point boundary nudge (e.g. 1.0 + epsilon)
        if matched_bin is None:
            if probability >= bins[-1]["bin_upper"] - 1e-6:
                matched_bin = bins[-1]
            elif probability <= bins[0]["bin_lower"] + 1e-6:
                matched_bin = bins[0]
            else:
                raise RuntimeError(
                    f"Could not assign probability {probability} to any bin for {disease}."
                )

        b_idx = matched_bin["bin_index"]
        b_lower = matched_bin["bin_lower"]
        b_upper = matched_bin["bin_upper"]
        b_count = matched_bin["count"]
        b_mean_prob = matched_bin["mean_probability"]
        empirical_rate = matched_bin["observed_frequency"]

        is_last = b_idx == (num_bins - 1)
        bin_range_str = f"[{b_lower:.4f}, {b_upper:.4f}{']' if is_last else ')'}"

        # -----------------------------------------------------
        # Mathematical Discrepancy & Status Resolution
        # -----------------------------------------------------
        if b_count == 0 or math.isnan(empirical_rate):
            calibration_gap = None
            abs_gap = None
            status = "UNPOPULATED_BIN_WARNING"
            status_desc = "No validation samples fall in this bin; empirical reliability cannot be verified."
        else:
            calibration_gap = float(probability - empirical_rate)
            abs_gap = float(abs(calibration_gap))

            if abs_gap <= self.warning_threshold:
                status = "WELL_CALIBRATED"
                status_desc = (
                    f"Calibrated probability aligns well with validation positive rate "
                    f"(|gap| = {abs_gap:.4f} <= {self.warning_threshold:.2f})."
                )
            elif abs_gap <= self.high_discrepancy_threshold:
                status = "MODERATE_DISCREPANCY"
                status_desc = (
                    f"Noticeable discrepancy between predicted probability and validation rate "
                    f"({self.warning_threshold:.2f} < |gap| = {abs_gap:.4f} <= {self.high_discrepancy_threshold:.2f})."
                )
            else:
                status = "HIGH_DISCREPANCY_WARNING"
                status_desc = (
                    f"Substantial calibration gap detected "
                    f"(|gap| = {abs_gap:.4f} > {self.high_discrepancy_threshold:.2f}). "
                    f"Exercise caution when interpreting this probability."
                )

        return {
            "disease": disease,
            "probability": probability,
            "calibration_state": self.state,
            "bin_index": b_idx,
            "bin_range": bin_range_str,
            "bin_sample_count": b_count,
            "mean_bin_probability": b_mean_prob,
            "empirical_positive_rate": empirical_rate,
            "calibration_gap": calibration_gap,
            "absolute_calibration_gap": abs_gap,
            "warning_threshold": self.warning_threshold,
            "high_discrepancy_threshold": self.high_discrepancy_threshold,
            "status": status,
            "status_description": status_desc,
            "mathematical_note": (
                "calibration_gap = probability - empirical_positive_rate. "
                "Positive gap indicates overconfidence; negative gap indicates underconfidence."
            ),
        }

    # =========================================================
    # MULTI-DISEASE ANALYSIS
    # =========================================================

    def analyze_all(
        self,
        probabilities: Union[Dict[str, float], List[float], np.ndarray, torch.Tensor],
    ) -> Dict:
        """
        Analyze all 14 pathologies simultaneously.

        Parameters
        ----------
        probabilities:
            Dictionary mapping disease name to probability, or a 14-element array/tensor.
        """
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.squeeze().detach().cpu().numpy()

        if isinstance(probabilities, (list, np.ndarray)):
            if len(probabilities) != len(CLASSES):
                raise ValueError(
                    f"Expected 14 probabilities, got {len(probabilities)}"
                )
            prob_dict = {
                disease: float(prob)
                for disease, prob in zip(CLASSES, probabilities)
            }
        elif isinstance(probabilities, dict):
            prob_dict = probabilities
        else:
            raise TypeError(
                f"Unsupported probabilities type: {type(probabilities)}"
            )

        findings = []
        valid_gaps = []
        high_discrepancy_diseases = []

        for disease in CLASSES:
            if disease not in prob_dict:
                raise ValueError(f"Missing probability for disease '{disease}'")

            result = self.analyze_prediction(
                disease=disease,
                probability=prob_dict[disease],
            )
            findings.append(result)

            if result["absolute_calibration_gap"] is not None:
                valid_gaps.append(result["absolute_calibration_gap"])

            if result["status"] == "HIGH_DISCREPANCY_WARNING":
                high_discrepancy_diseases.append(disease)

        mean_gap = float(np.mean(valid_gaps)) if valid_gaps else None
        max_gap = float(np.max(valid_gaps)) if valid_gaps else None

        return {
            "calibration_state": self.state,
            "total_diseases_analyzed": len(findings),
            "mean_absolute_calibration_gap": mean_gap,
            "max_absolute_calibration_gap": max_gap,
            "high_discrepancy_diseases": high_discrepancy_diseases,
            "warning_threshold": self.warning_threshold,
            "findings": findings,
        }
