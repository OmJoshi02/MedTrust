import json
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.calibration.calibration_analyzer import CalibrationAnalyzer
from src.dataset import CLASSES
from src.explainability.gradcam import GradCAM
from src.explainability.reliability import ExplanationReliabilityAnalyzer
from src.model import create_model
from src.transformer import test_transform


class TestIntegrationModules6And9(unittest.TestCase):
    """
    End-to-End integration test connecting:
    X-ray -> Preprocessing -> Classifier -> Grad-CAM ->
    Explanation Reliability Analyzer (Mod 6) -> Calibrated prediction ->
    Calibration Analyzer (Mod 9) -> Structured Diagnostic Output
    """

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.checkpoint_path = Path("checkpoints/best_model.pt")
        cls.temperature_path = Path("outputs/calibration/temperature.json")
        cls.test_image_path = Path("test_images/disease.png")

        assert cls.checkpoint_path.exists(), f"Checkpoint missing: {cls.checkpoint_path}"
        assert cls.temperature_path.exists(), f"Temperature file missing: {cls.temperature_path}"
        assert cls.test_image_path.exists(), f"Test image missing: {cls.test_image_path}"

        # Load model
        ckpt = torch.load(cls.checkpoint_path, map_location=cls.device, weights_only=False)
        cls.model = create_model(num_classes=14, pretrained=False)
        cls.model.load_state_dict(ckpt["model_state_dict"])
        cls.model.to(cls.device).eval()

        # Load temperature scaling parameters
        with open(cls.temperature_path, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
        cls.temperatures = np.array(
            [float(temp_data["temperatures"][d]) for d in CLASSES],
            dtype=np.float32,
        )

    def test_end_to_end_flow(self):
        # 1. Input & Preprocessing
        image = Image.open(self.test_image_path).convert("RGB")
        image_tensor = test_transform(image).unsqueeze(0).to(self.device)

        # 2. Classifier Prediction
        with torch.inference_mode():
            logits = self.model(image_tensor)
            raw_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

            temp_tensor = torch.tensor(
                self.temperatures,
                dtype=logits.dtype,
                device=self.device,
            ).unsqueeze(0)
            calibrated_logits = logits / temp_tensor
            calibrated_probs = torch.sigmoid(calibrated_logits).squeeze(0).cpu().numpy()

        top_disease_idx = int(np.argmax(raw_probs))
        top_disease = CLASSES[top_disease_idx]

        # 3. Grad-CAM Generation
        target_layer = self.model.features.norm5
        with GradCAM(model=self.model, target_layer=target_layer) as gradcam:
            cam, _ = gradcam.generate(input_tensor=image_tensor, class_index=top_disease_idx)
            cam_np = cam.cpu().numpy()

        self.assertEqual(cam_np.shape, (224, 224))
        self.assertTrue(0.0 <= cam_np.min() <= cam_np.max() <= 1.0)

        # 4. Explanation Reliability Analyzer (Module 6)
        reliability_analyzer = ExplanationReliabilityAnalyzer(
            model=self.model,
            target_layer=target_layer,
            device=self.device,
            random_seed=42,
            occlusion_fraction=0.20,
        )
        reliability_result = reliability_analyzer.analyze(
            image_input=image,
            class_index=top_disease_idx,
        )

        self.assertEqual(reliability_result["target_disease"], top_disease)
        self.assertIn(reliability_result["overall_stability"], ("HIGH", "MODERATE", "LOW"))
        self.assertTrue(-1.0 <= reliability_result["mean_pearson_correlation"] <= 1.0)
        self.assertTrue(-1.0 <= reliability_result["mean_spearman_correlation"] <= 1.0)
        self.assertTrue(0.0 <= reliability_result["mean_absolute_difference"] <= 1.0)
        self.assertTrue(0.0 <= reliability_result["mean_region_iou"] <= 1.0)
        self.assertIn("fidelity_probability_drop", reliability_result)

        # 5. Calibration Analyzer (Module 9)
        calibration_analyzer = CalibrationAnalyzer(state="after")

        # Single-disease diagnostic for the top predicted finding
        top_calib_diagnostic = calibration_analyzer.analyze_prediction(
            disease=top_disease,
            probability=calibrated_probs[top_disease_idx],
        )

        self.assertEqual(top_calib_diagnostic["disease"], top_disease)
        self.assertIn("bin_index", top_calib_diagnostic)
        self.assertIn("calibration_gap", top_calib_diagnostic)
        self.assertIn(
            top_calib_diagnostic["status"],
            ("WELL_CALIBRATED", "MODERATE_DISCREPANCY", "HIGH_DISCREPANCY_WARNING", "UNPOPULATED_BIN_WARNING"),
        )

        # Multi-disease diagnostic across all 14 pathologies
        all_calib_diagnostic = calibration_analyzer.analyze_all(calibrated_probs)
        self.assertEqual(all_calib_diagnostic["total_diseases_analyzed"], 14)
        self.assertIsNotNone(all_calib_diagnostic["mean_absolute_calibration_gap"])

        # 6. Combined Structured Diagnostic Output verification
        diagnostic_record = {
            "image": str(self.test_image_path),
            "top_predicted_disease": top_disease,
            "raw_probability": float(raw_probs[top_disease_idx]),
            "calibrated_probability": float(calibrated_probs[top_disease_idx]),
            "explanation_reliability": {
                "stability_band": reliability_result["overall_stability"],
                "mean_pearson": reliability_result["mean_pearson_correlation"],
                "mean_spearman": reliability_result["mean_spearman_correlation"],
                "mean_mad": reliability_result["mean_absolute_difference"],
                "mean_iou": reliability_result["mean_region_iou"],
                "fidelity_drop": reliability_result["fidelity_probability_drop"],
                "fidelity_verified": reliability_result["fidelity_verified"],
            },
            "calibration_diagnostic": {
                "bin_range": top_calib_diagnostic["bin_range"],
                "empirical_positive_rate": top_calib_diagnostic["empirical_positive_rate"],
                "calibration_gap": top_calib_diagnostic["calibration_gap"],
                "status": top_calib_diagnostic["status"],
            },
        }

        self.assertIsInstance(diagnostic_record["explanation_reliability"]["mean_pearson"], float)
        self.assertIsInstance(diagnostic_record["calibration_diagnostic"]["calibration_gap"], float)


if __name__ == "__main__":
    unittest.main()
