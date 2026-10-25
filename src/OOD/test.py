"""
Unit Tests for MedTrust OOD Detection Module (src/OOD/test.py)

Validates:
    - Mathematical properties of MSP, Energy, and Mahalanobis scoring
    - Feature statistics estimation and Mahalanobis distance
    - Conservative categorization (in-distribution-like, borderline/uncertain, out-of-distribution-like)
    - Finite numerical checks (NaN / Inf protection)
    - Shape validation and error handling
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.OOD.common import (
    validate_finite_array,
    validate_ood_scores,
    validate_class_order,
    ensure_dir,
    get_device,
)
from src.OOD.scores import (
    msp_score,
    energy_score,
    mahalanobis_score,
    fit_mahalanobis_statistics,
    score_all_methods,
)
from src.OOD.detector import OODDetector, OODResult


class TestOODScores(unittest.TestCase):
    """Test suite for mathematical OOD scoring functions."""

    def test_msp_score_bounds_and_behavior(self):
        # Confident prediction (high positive logit) -> high probability -> low OOD score
        confident_logits = np.array([[5.0, -2.0, -3.0]])
        # Diffuse / uncertain prediction (all low logits) -> low probability -> high OOD score
        uncertain_logits = np.array([[-2.0, -2.0, -2.0]])

        confident_score = msp_score(confident_logits)[0]
        uncertain_score = msp_score(uncertain_logits)[0]

        self.assertGreaterEqual(confident_score, 0.0)
        self.assertLessEqual(confident_score, 1.0)
        self.assertGreater(uncertain_score, confident_score)

    def test_msp_rejects_nan_and_bad_dim(self):
        with self.assertRaises(ValueError):
            msp_score(np.array([1.0, 2.0]))  # 1D instead of 2D
        with self.assertRaises(ValueError):
            msp_score(np.array([[1.0, np.nan]]))

    def test_energy_score_behavior(self):
        confident_logits = np.array([[10.0, 0.0, 0.0]])
        uncertain_logits = np.array([[0.0, 0.0, 0.0]])

        score_conf = energy_score(confident_logits, temperature=1.0)[0]
        score_unc = energy_score(uncertain_logits, temperature=1.0)[0]

        # Higher energy score indicates more OOD-like
        self.assertGreater(score_unc, score_conf)

    def test_energy_score_temperature_scaling(self):
        logits = np.array([[3.0, 1.0, -1.0]])
        s_t1 = energy_score(logits, temperature=1.0)[0]
        s_t2 = energy_score(logits, temperature=2.0)[0]
        self.assertTrue(np.isfinite(s_t1) and np.isfinite(s_t2))

    def test_fit_and_calculate_mahalanobis(self):
        # Generate synthetic reference features [N=100, D=8]
        np.random.seed(42)
        features = np.random.randn(100, 8) + 2.0

        mean, safe_variance = fit_mahalanobis_statistics(features)

        self.assertEqual(mean.shape, (8,))
        self.assertEqual(safe_variance.shape, (8,))
        self.assertTrue(np.all(safe_variance > 0))

        # Test point at exact mean should have distance close to 0
        at_mean = mean.reshape(1, 8)
        dist_at_mean = mahalanobis_score(at_mean, mean, safe_variance)[0]
        self.assertAlmostEqual(dist_at_mean, 0.0, places=4)

        # Distant outlier should have large distance
        distant = (mean + 10.0).reshape(1, 8)
        dist_distant = mahalanobis_score(distant, mean, safe_variance)[0]
        self.assertGreater(dist_distant, dist_at_mean)

    def test_score_all_methods(self):
        logits = np.array([[2.0, -1.0], [0.5, 0.5]])
        results = score_all_methods(logits)
        self.assertIn("msp", results)
        self.assertIn("energy", results)
        self.assertEqual(len(results["msp"]), 2)
        self.assertEqual(len(results["energy"]), 2)


class TestOODDetector(unittest.TestCase):
    """Test suite for OODDetector orchestration and conservative categorization."""

    def test_detector_initialization(self):
        detector = OODDetector(method="energy", threshold=-2.0)
        self.assertEqual(detector.method, "energy")
        self.assertEqual(detector.threshold, -2.0)

        with self.assertRaises(ValueError):
            OODDetector(method="unsupported_method")

    def test_conservative_categorization(self):
        # Given threshold = 10.0, margin = 0.20
        # Lower bound = 8.0, Upper bound = 12.0
        detector = OODDetector(method="msp", threshold=0.50, margin=0.20)

        # Score well below threshold -> in-distribution-like
        status, is_ood, _ = detector.categorize(0.20)
        self.assertEqual(status, "in-distribution-like")
        self.assertFalse(is_ood)

        # Score near threshold -> borderline/uncertain
        status, is_ood, _ = detector.categorize(0.50)
        self.assertEqual(status, "borderline/uncertain")
        self.assertFalse(is_ood)

        # Score well above threshold -> out-of-distribution-like
        status, is_ood, _ = detector.categorize(0.90)
        self.assertEqual(status, "out-of-distribution-like")
        self.assertTrue(is_ood)

    def test_detect_from_logits(self):
        detector = OODDetector(method="energy", threshold=0.0)
        dummy_logits = np.array([[3.0, 1.0, 0.0, -1.0]])

        result = detector.detect(logits=dummy_logits)
        self.assertIsInstance(result, OODResult)
        self.assertIn(result.status, {"in-distribution-like", "borderline/uncertain", "out-of-distribution-like"})
        self.assertTrue(np.isfinite(result.score))

        # Format report should render without error
        report_str = result.format_report()
        self.assertIn("MEDTRUST - OUT-OF-DISTRIBUTION", report_str)


class TestCommonUtilities(unittest.TestCase):
    """Test suite for validation and utility functions."""

    def test_finite_checks(self):
        valid = np.array([1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(validate_finite_array(valid), valid))

        with self.assertRaises(ValueError):
            validate_finite_array(np.array([1.0, np.nan]))

        with self.assertRaises(ValueError):
            validate_finite_array(np.array([1.0, np.inf]))

    def test_validate_class_order(self):
        classes = ["A", "B", "C"]
        validate_class_order(classes, ["A", "B", "C"])  # Should pass

        with self.assertRaises(ValueError):
            validate_class_order(["B", "A", "C"], ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
