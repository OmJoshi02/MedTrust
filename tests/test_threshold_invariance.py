"""
Unit test verifying Dual Threshold Mapping invariance across all 14 classes.
"""

import unittest
from PIL import Image
from pathlib import Path
import numpy as np

from backend.engine import MedTrustBackendEngine
from src.dataset import CLASSES


class TestDualThresholdInvariance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.engine = MedTrustBackendEngine()
        cls.demo_path = Path("test_images/disease.png")
        cls.image = Image.open(cls.demo_path)

    def test_mathematical_invariance_all_classes(self):
        res = self.engine.run_prediction(self.image)
        findings = res["findings"]
        self.assertEqual(len(findings), len(CLASSES))

        for f in findings:
            name = f["disease"]
            p_raw = f["raw_probability"]
            p_cal = f["calibrated_probability"]
            t_raw = f["raw_threshold"]
            t_cal = f["calibrated_threshold"]
            dec_raw = p_raw >= t_raw
            dec_cal = p_cal >= t_cal

            # Invariance assertion: Decision in raw space MUST equal decision in calibrated space
            self.assertEqual(
                dec_raw,
                dec_cal,
                f"Mismatch for {name}: raw {p_raw:.4f} >= {t_raw:.4f} ({dec_raw}) vs cal {p_cal:.4f} >= {t_cal:.4f} ({dec_cal})"
            )
            self.assertEqual(
                f["predicted_positive"],
                dec_raw,
                f"Flag mismatch for {name}"
            )

    def test_effusion_specific_numbers(self):
        res = self.engine.run_prediction(self.image)
        effusion = next(f for f in res["findings"] if f["disease"] == "Effusion")
        
        # Verify calibrated threshold for Effusion is ~57.7%
        self.assertAlmostEqual(effusion["raw_threshold"], 0.6280, delta=0.005)
        self.assertAlmostEqual(effusion["calibrated_threshold"], 0.5766, delta=0.005)
        self.assertAlmostEqual(effusion["threshold"], effusion["calibrated_threshold"], delta=0.001)

        # In this image, raw prob is ~68.8% and cal prob is ~61.4%
        # Both are strictly >= their corresponding threshold!
        self.assertTrue(effusion["raw_probability"] >= effusion["raw_threshold"])
        self.assertTrue(effusion["calibrated_probability"] >= effusion["calibrated_threshold"])
        self.assertTrue(effusion["predicted_positive"])
        self.assertEqual(effusion["decision"], "POSITIVE")


if __name__ == "__main__":
    unittest.main()

