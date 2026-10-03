import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.calibration.calibration_analyzer import CalibrationAnalyzer
from src.dataset import CLASSES


class TestCalibrationAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.real_bins_path = Path("outputs/calibration/reliability/validation_reliability_bins.csv")
        assert cls.real_bins_path.exists(), f"Real bins CSV missing at {cls.real_bins_path}"

    def test_init_and_threshold_validation(self):
        """Test parameter validation during initialization."""
        with self.assertRaises(ValueError):
            CalibrationAnalyzer(state="invalid_state")

        with self.assertRaises(ValueError):
            # warning_threshold >= high_discrepancy_threshold
            CalibrationAnalyzer(warning_threshold=0.25, high_discrepancy_threshold=0.10)

        with self.assertRaises(ValueError):
            # Negative threshold
            CalibrationAnalyzer(warning_threshold=-0.05)

    def test_missing_and_malformed_csv(self):
        """Test handling of non-existent and malformed CSV files."""
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            CalibrationAnalyzer(bins_csv_path="non_existent_file.csv")

        # Malformed CSV missing required column
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            tmp.write("split,disease,state,bin_index\n")
            tmp.write("validation,Atelectasis,after,0\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError):
                CalibrationAnalyzer(bins_csv_path=tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_exact_boundary_conditions(self):
        """Test probability exactly 0.0, 1.0, and exactly on a bin boundary."""
        analyzer = CalibrationAnalyzer(self.real_bins_path, state="after")

        # Exactly 0.0 -> Must fall in bin 0
        res_0 = analyzer.analyze_prediction("Pneumonia", 0.0)
        self.assertEqual(res_0["bin_index"], 0)
        self.assertEqual(res_0["probability"], 0.0)
        self.assertIsNotNone(res_0["status"])

        # Exactly 1.0 -> Must fall in the last bin (bin 14)
        res_1 = analyzer.analyze_prediction("Pneumonia", 1.0)
        self.assertEqual(res_1["bin_index"], 14)
        self.assertEqual(res_1["probability"], 1.0)
        self.assertIsNotNone(res_1["status"])

        # Exact bin boundary (e.g. 0.20): should fall into bin where bin_lower == 0.20
        # By convention [bin_lower, bin_upper), 0.20 is lower bound of bin 3 (since 0.06666 * 3 = 0.20)
        res_bound = analyzer.analyze_prediction("Effusion", 0.20)
        self.assertTrue(res_bound["bin_index"] >= 1)

    def test_invalid_probabilities_and_edge_values(self):
        """Test that invalid values (NaN, Inf, negative, >1.0, strings) raise ValueErrors/TypeErrors."""
        analyzer = CalibrationAnalyzer(self.real_bins_path, state="after")

        with self.assertRaises(ValueError):
            analyzer.analyze_prediction("Effusion", -0.01)

        with self.assertRaises(ValueError):
            analyzer.analyze_prediction("Effusion", 1.01)

        with self.assertRaises(ValueError):
            analyzer.analyze_prediction("Effusion", float("nan"))

        with self.assertRaises(ValueError):
            analyzer.analyze_prediction("Effusion", float("inf"))

        with self.assertRaises(TypeError):
            analyzer.analyze_prediction("Effusion", "high")

    def test_unsupported_disease(self):
        """Test that unknown disease names raise a ValueError."""
        analyzer = CalibrationAnalyzer(self.real_bins_path, state="after")
        with self.assertRaises(ValueError):
            analyzer.analyze_prediction("COVID-19", 0.5)

    def test_mathematical_gap_and_status(self):
        """Verify that calibration_gap = probability - empirical_rate and statuses match thresholds."""
        analyzer = CalibrationAnalyzer(
            self.real_bins_path,
            state="after",
            warning_threshold=0.10,
            high_discrepancy_threshold=0.20,
        )

        res = analyzer.analyze_prediction("Atelectasis", 0.75)
        prob = res["probability"]
        emp = res["empirical_positive_rate"]
        gap = res["calibration_gap"]
        abs_gap = res["absolute_calibration_gap"]

        self.assertAlmostEqual(gap, prob - emp, places=5)
        self.assertAlmostEqual(abs_gap, abs(gap), places=5)

        if abs_gap <= 0.10:
            self.assertEqual(res["status"], "WELL_CALIBRATED")
        elif abs_gap <= 0.20:
            self.assertEqual(res["status"], "MODERATE_DISCREPANCY")
        else:
            self.assertEqual(res["status"], "HIGH_DISCREPANCY_WARNING")

    def test_empty_or_nan_bin_handling(self):
        """Test synthetic scenario where a bin has zero count or NaN observed frequency."""
        # Create a small valid CSV with one bin having count=0 and observed_frequency=NaN
        rows = []
        for d in CLASSES:
            for b_idx in range(15):
                lower = b_idx / 15.0
                upper = (b_idx + 1) / 15.0
                if d == "Hernia" and b_idx == 14:
                    rows.append({
                        "split": "validation",
                        "disease": d,
                        "state": "after",
                        "bin_index": b_idx,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "count": 0,
                        "mean_probability": 0.95,
                        "observed_frequency": float("nan"),
                    })
                else:
                    rows.append({
                        "split": "validation",
                        "disease": d,
                        "state": "after",
                        "bin_index": b_idx,
                        "bin_lower": lower,
                        "bin_upper": upper,
                        "count": 50,
                        "mean_probability": lower + 0.03,
                        "observed_frequency": lower + 0.02,
                    })

        synth_df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
            synth_df.to_csv(tmp.name, index=False)
            tmp_path = Path(tmp.name)

        try:
            synth_analyzer = CalibrationAnalyzer(bins_csv_path=tmp_path, state="after")
            res_nan = synth_analyzer.analyze_prediction("Hernia", 0.98)
            self.assertEqual(res_nan["status"], "UNPOPULATED_BIN_WARNING")
            self.assertIsNone(res_nan["calibration_gap"])
            self.assertIsNone(res_nan["absolute_calibration_gap"])
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_analyze_all_inputs(self):
        """Test analyze_all with dictionary, numpy array, and PyTorch tensor."""
        analyzer = CalibrationAnalyzer(self.real_bins_path, state="after")

        # 1. NumPy array
        np_probs = np.full(14, 0.35, dtype=np.float32)
        res_np = analyzer.analyze_all(np_probs)
        self.assertEqual(res_np["total_diseases_analyzed"], 14)
        self.assertIsNotNone(res_np["mean_absolute_calibration_gap"])

        # 2. PyTorch tensor
        torch_probs = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.2, 0.3, 0.4, 0.5, 0.6, 0.1])
        res_torch = analyzer.analyze_all(torch_probs)
        self.assertEqual(len(res_torch["findings"]), 14)

        # 3. Dictionary
        dict_probs = {d: 0.15 for d in CLASSES}
        res_dict = analyzer.analyze_all(dict_probs)
        self.assertEqual(len(res_dict["findings"]), 14)


if __name__ == "__main__":
    unittest.main()
