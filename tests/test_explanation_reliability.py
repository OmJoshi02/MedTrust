import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.dataset import CLASSES
from src.explainability.reliability import ExplanationReliabilityAnalyzer
from src.model import create_model


class MockConvModel(nn.Module):
    """
    Minimal convolutional model with a features block and norm5 layer
    to test ExplanationReliabilityAnalyzer rapidly without requiring GPU/DenseNet.
    """

    def __init__(self, num_classes=14):
        super().__init__()
        self.features = nn.Sequential()
        self.features.add_module("conv1", nn.Conv2d(3, 16, kernel_size=3, padding=1))
        self.features.add_module("relu1", nn.ReLU(inplace=True))
        self.features.add_module("norm5", nn.BatchNorm2d(16))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.features(x)
        pooled = self.pool(feat)
        flat = torch.flatten(pooled, 1)
        return self.classifier(flat)


class TestExplanationReliabilityAnalyzer(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.mock_model = MockConvModel(num_classes=14).eval()
        self.sample_tensor = torch.randn(1, 3, 224, 224)
        self.sample_image = Image.new("RGB", (224, 224), color=(128, 128, 128))

    def test_parameter_validation(self):
        """Test that invalid parameters raise descriptive ValueErrors."""
        with self.assertRaises(ValueError):
            ExplanationReliabilityAnalyzer(self.mock_model, occlusion_fraction=0.0)

        with self.assertRaises(ValueError):
            ExplanationReliabilityAnalyzer(self.mock_model, occlusion_fraction=1.5)

        with self.assertRaises(ValueError):
            ExplanationReliabilityAnalyzer(self.mock_model, brightness_factor=-0.5)

        with self.assertRaises(ValueError):
            ExplanationReliabilityAnalyzer(self.mock_model, high_threshold=0.5, moderate_threshold=0.8)

    def test_metric_computations_and_bounds(self):
        """Verify metric bounds for Pearson, Spearman, MAD, and IoU."""
        cam1 = np.random.rand(50, 50).astype(np.float32)
        cam2 = np.random.rand(50, 50).astype(np.float32)

        # Same CAM should yield 1.0 correlation, 0.0 MAD, 1.0 IoU
        p_self = ExplanationReliabilityAnalyzer.compute_pearson_correlation(cam1, cam1)
        s_self = ExplanationReliabilityAnalyzer.compute_spearman_correlation(cam1, cam1)
        mad_self = ExplanationReliabilityAnalyzer.compute_mean_absolute_difference(cam1, cam1)
        iou_self = ExplanationReliabilityAnalyzer.compute_region_iou(cam1, cam1, top_fraction=0.20)

        self.assertAlmostEqual(p_self, 1.0, places=5)
        self.assertAlmostEqual(s_self, 1.0, places=5)
        self.assertAlmostEqual(mad_self, 0.0, places=5)
        self.assertAlmostEqual(iou_self, 1.0, places=5)

        # Different CAMs should stay within bounds
        p_diff = ExplanationReliabilityAnalyzer.compute_pearson_correlation(cam1, cam2)
        s_diff = ExplanationReliabilityAnalyzer.compute_spearman_correlation(cam1, cam2)
        mad_diff = ExplanationReliabilityAnalyzer.compute_mean_absolute_difference(cam1, cam2)
        iou_diff = ExplanationReliabilityAnalyzer.compute_region_iou(cam1, cam2, top_fraction=0.20)

        self.assertTrue(-1.0 <= p_diff <= 1.0)
        self.assertTrue(-1.0 <= s_diff <= 1.0)
        self.assertTrue(0.0 <= mad_diff <= 1.0)
        self.assertTrue(0.0 <= iou_diff <= 1.0)

    def test_flat_cam_edge_case(self):
        """Verify division-by-zero protection on constant/flat CAMs."""
        flat_cam1 = np.zeros((30, 30), dtype=np.float32)
        flat_cam2 = np.zeros((30, 30), dtype=np.float32)
        non_flat = np.random.rand(30, 30).astype(np.float32)

        # Two identical flat CAMs should be 1.0, not NaN or crash
        p_flat = ExplanationReliabilityAnalyzer.compute_pearson_correlation(flat_cam1, flat_cam2)
        s_flat = ExplanationReliabilityAnalyzer.compute_spearman_correlation(flat_cam1, flat_cam2)
        self.assertEqual(p_flat, 1.0)
        self.assertEqual(s_flat, 1.0)

        # One flat, one non-flat should be 0.0, not NaN
        p_mixed = ExplanationReliabilityAnalyzer.compute_pearson_correlation(flat_cam1, non_flat)
        s_mixed = ExplanationReliabilityAnalyzer.compute_spearman_correlation(flat_cam1, non_flat)
        self.assertEqual(p_mixed, 0.0)
        self.assertEqual(s_mixed, 0.0)

    def test_reproducibility_with_seed(self):
        """Verify that identical random_seed yields bitwise identical perturbation metrics."""
        analyzer1 = ExplanationReliabilityAnalyzer(self.mock_model, random_seed=42)
        analyzer2 = ExplanationReliabilityAnalyzer(self.mock_model, random_seed=42)

        res1 = analyzer1.analyze(self.sample_image, class_index=0)
        res2 = analyzer2.analyze(self.sample_image, class_index=0)

        self.assertAlmostEqual(res1["mean_pearson_correlation"], res2["mean_pearson_correlation"], places=6)
        self.assertAlmostEqual(res1["mean_spearman_correlation"], res2["mean_spearman_correlation"], places=6)
        self.assertAlmostEqual(res1["mean_absolute_difference"], res2["mean_absolute_difference"], places=6)
        self.assertAlmostEqual(res1["mean_region_iou"], res2["mean_region_iou"], places=6)
        self.assertAlmostEqual(res1["fidelity_probability_drop"], res2["fidelity_probability_drop"], places=6)

    def test_model_weights_not_modified(self):
        """Ensure analyzer is strictly diagnostic and does not mutate model parameters."""
        initial_params = [p.clone().detach() for p in self.mock_model.parameters()]

        analyzer = ExplanationReliabilityAnalyzer(self.mock_model, random_seed=123)
        _ = analyzer.analyze(self.sample_image, class_index=3)

        for p_init, p_after in zip(initial_params, self.mock_model.parameters()):
            self.assertTrue(torch.equal(p_init, p_after), "Model parameter was mutated during analysis!")

    def test_fidelity_occlusion(self):
        """Test region occlusion fidelity computation."""
        analyzer = ExplanationReliabilityAnalyzer(self.mock_model, occlusion_fraction=0.25)
        res = analyzer.analyze(self.sample_tensor, class_index=2)

        fidelity = res["fidelity"]
        self.assertIn("fidelity_probability_before", fidelity)
        self.assertIn("fidelity_probability_after", fidelity)
        self.assertIn("fidelity_probability_drop", fidelity)
        self.assertEqual(fidelity["occlusion_fraction"], 0.25)
        self.assertAlmostEqual(
            fidelity["fidelity_probability_drop"],
            fidelity["fidelity_probability_before"] - fidelity["fidelity_probability_after"],
            places=5,
        )

    def test_invalid_class_index(self):
        """Test boundary checking on class index."""
        analyzer = ExplanationReliabilityAnalyzer(self.mock_model)
        with self.assertRaises(IndexError):
            analyzer.analyze(self.sample_image, class_index=-1)
        with self.assertRaises(IndexError):
            analyzer.analyze(self.sample_image, class_index=14)

    def test_cpu_and_gpu_compatibility(self):
        """Verify device compatibility."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MockConvModel().to(device).eval()
        analyzer = ExplanationReliabilityAnalyzer(model, device=device)
        res = analyzer.analyze(self.sample_image, class_index=6)
        self.assertIn("overall_stability", res)
        self.assertIn("fidelity", res)


if __name__ == "__main__":
    unittest.main()
