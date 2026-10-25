"""
MedTrust V1 — Unit Tests for Module 5: Technical Image Quality Analysis
(tests/test_image_quality.py)
"""

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.quality import (
    DEFAULT_REFERENCE_PATH,
    WEIGHTS,
    QualityReference,
    QualityResult,
    WeightedImageQualityAnalyzer,
    calculate_brightness,
    calculate_clipping,
    calculate_contrast,
    calculate_noise,
    calculate_raw_metrics,
    calculate_resolution,
    calculate_sharpness,
    create_default_reference,
    determine_status,
    generate_warnings,
    load_grayscale_image,
    normalize_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE_PATH = PROJECT_ROOT / "test_images" / "disease.png"


class TestImageQualityMetrics(unittest.TestCase):
    """Test mathematical and operational properties of image quality metrics."""

    def test_weights_configuration(self):
        expected_metrics = {"sharpness", "contrast", "brightness", "clipping", "noise", "resolution"}
        self.assertEqual(set(WEIGHTS.keys()), expected_metrics)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=6)

    def test_reference_validation(self):
        ref = create_default_reference()
        self.assertGreater(ref.sharpness_high, ref.sharpness_low)
        self.assertGreater(ref.contrast_high, ref.contrast_low)
        self.assertGreater(ref.clipping_high, ref.clipping_low)
        self.assertGreater(ref.noise_high, ref.noise_low)
        self.assertGreater(ref.resolution_high, ref.resolution_low)
        self.assertTrue(0 <= ref.brightness_target <= 255)

        # Invalid reference ranges should raise ValueError
        with self.assertRaises(ValueError):
            QualityReference(sharpness_low=100.0, sharpness_high=50.0)

    def test_raw_metrics_synthetic(self):
        # Uniform gray image (128)
        uniform = np.full((100, 100), 128, dtype=np.uint8)
        raw = calculate_raw_metrics(uniform)

        self.assertAlmostEqual(raw["sharpness"], 0.0, places=4)
        self.assertAlmostEqual(raw["contrast"], 0.0, places=4)
        self.assertAlmostEqual(raw["brightness"], 128.0, places=4)
        self.assertAlmostEqual(raw["clipping"], 0.0, places=4)
        self.assertAlmostEqual(raw["noise"], 0.0, places=4)
        self.assertEqual(raw["resolution"], 100.0)

    def test_clipping_metric(self):
        # 10x10 image with 10 pure black and 10 pure white pixels
        img = np.full((10, 10), 128, dtype=np.uint8)
        img[0, :] = 0    # 10 black pixels
        img[1, :] = 255  # 10 white pixels
        clipping = calculate_clipping(img)
        self.assertAlmostEqual(clipping, 20.0, places=2)

    def test_normalization_bounds(self):
        ref = create_default_reference()
        # Extreme raw metrics
        raw = {
            "sharpness": 9999.0,
            "contrast": -50.0,
            "brightness": 500.0,
            "clipping": 100.0,
            "noise": -10.0,
            "resolution": 5000.0,
        }
        norm = normalize_metrics(raw, ref)
        for k, v in norm.items():
            self.assertGreaterEqual(v, 0.0, f"Metric {k} was below 0.0: {v}")
            self.assertLessEqual(v, 1.0, f"Metric {k} was above 1.0: {v}")

    def test_polymorphic_loader(self):
        # 1. From PIL Image
        pil_img = Image.new("L", (120, 80), color=150)
        arr1 = load_grayscale_image(pil_img)
        self.assertEqual(arr1.shape, (80, 120))
        self.assertEqual(arr1.dtype, np.uint8)

        # 2. From 3D NumPy RGB array
        rgb_arr = np.ones((60, 60, 3), dtype=np.uint8) * 100
        arr2 = load_grayscale_image(rgb_arr)
        self.assertEqual(arr2.shape, (60, 60))
        self.assertEqual(arr2.dtype, np.uint8)

        # 3. From invalid path
        with self.assertRaises(FileNotFoundError):
            load_grayscale_image("nonexistent_image_12345.png")

        # 4. From empty array
        with self.assertRaises(ValueError):
            load_grayscale_image(np.array([]))

    def test_pathology_safeguard_warning(self):
        # Case where sharpness is low, but other parameters are healthy
        raw = {
            "sharpness": 5.0,     # Low
            "contrast": 60.0,     # Healthy
            "brightness": 160.0,  # Optimal
            "clipping": 2.0,      # Low
            "noise": 2.0,         # Low
            "resolution": 1024.0, # High
        }
        ref = create_default_reference()
        norm = normalize_metrics(raw, ref)
        warnings = generate_warnings(raw, norm)

        safeguard_found = any("pathology" in w.lower() for w in warnings)
        self.assertTrue(safeguard_found, "Expected pathology safeguard warning when only sharpness is low.")

    def test_status_determination(self):
        self.assertEqual(determine_status(75.0, threshold=60.0), "ACCEPTED")
        self.assertEqual(determine_status(60.0, threshold=60.0), "ACCEPTED")
        self.assertEqual(determine_status(59.9, threshold=60.0), "REJECTED")

    def test_analyzer_on_sample_image(self):
        if not SAMPLE_IMAGE_PATH.exists():
            self.skipTest(f"Sample image not found at {SAMPLE_IMAGE_PATH}")

        analyzer = WeightedImageQualityAnalyzer()
        result = analyzer.analyze(SAMPLE_IMAGE_PATH)

        self.assertIsInstance(result, QualityResult)
        self.assertTrue(0.0 <= result.quality_score <= 100.0)
        self.assertIn(result.status, ["ACCEPTED", "REJECTED"])

        d = result.to_dict()
        self.assertIn("quality_score", d)
        self.assertIn("composite_score", d)
        self.assertIn("quality_status", d)
        self.assertIn("suitable", d)
        self.assertIn("warnings", d)


if __name__ == "__main__":
    unittest.main()

