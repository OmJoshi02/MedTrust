import unittest
import json
import numpy as np
import torch
from backend.server import format_analysis_for_response, load_calibration_metadata
from backend.engine import MedTrustBackendEngine

class TestAuditFixes(unittest.TestCase):
    def setUp(self):
        self.cal_meta = load_calibration_metadata()

    def test_calibration_comparative_metrics_present(self):
        """Verify that before-vs-after comparative metrics are correctly loaded and verified."""
        self.assertIn("comparative_metrics", self.cal_meta)
        comp = self.cal_meta["comparative_metrics"]
        self.assertIn("nll", comp)
        self.assertIn("brier", comp)
        self.assertIn("auroc", comp)
        self.assertIn("ece", comp)
        
        # Verify proper scoring rules improved (decreased)
        self.assertLess(comp["nll"]["after"], comp["nll"]["before"])
        self.assertLess(comp["brier"]["after"], comp["brier"]["before"])
        # AUROC preserved
        self.assertAlmostEqual(comp["auroc"]["before"], comp["auroc"]["after"], places=4)
        # Dual bins present
        self.assertIn("bins", self.cal_meta)
        self.assertIn("before_bins", self.cal_meta)
        self.assertGreater(len(self.cal_meta["bins"]), 0)
        self.assertGreater(len(self.cal_meta["before_bins"]), 0)

    def test_format_analysis_space_isolation(self):
        """Verify space isolation in format_analysis_for_response."""
        engine_result = {
            "prediction": {
                "findings": [
                    {
                        "disease": "Effusion",
                        "raw_probability": 0.688,
                        "calibrated_probability": 0.614,
                        "raw_threshold": 0.628,
                        "calibrated_threshold": 0.577,
                        "threshold": 0.628,
                        "raw_predicted_positive": True,
                        "calibrated_predicted_positive": True,
                        "decision": "POSITIVE"
                    },
                    {
                        "disease": "Atelectasis",
                        "raw_probability": 0.400,
                        "calibrated_probability": 0.380,
                        "raw_threshold": 0.500,
                        "calibrated_threshold": 0.450,
                        "threshold": 0.500,
                        "raw_predicted_positive": False,
                        "calibrated_predicted_positive": False,
                        "decision": "NEGATIVE"
                    }
                ]
            },
            "ood": {
                "prediction": "IN_DISTRIBUTION",
                "method_results": {
                    "mahalanobis": {"score": 636.927, "threshold": 843.067, "is_ood": False, "status": "IN_DISTRIBUTION"},
                    "energy": {"score": -12.4, "threshold": -5.1, "is_ood": False, "status": "IN_DISTRIBUTION"},
                    "msp": {"score": 0.85, "threshold": 0.40, "is_ood": False, "status": "IN_DISTRIBUTION"}
                }
            },
            "explainability": {
                "top_disease": "Effusion",
                "target_layer": "features.norm5 (DenseNet-121 final conv, 7x7 spatial resolution)",
                "stability_spearman": 0.999,
                "faithfulness_drop": 0.516,
                "artifacts": [
                    {
                        "name": "heatmap",
                        "label": "Grad-CAM Heatmap",
                        "caption": "DenseNet-121 features.norm5 (7x7 spatial receptive field, smoothed to 224x224)."
                    }
                ]
            },
            "image_size_bytes": 254000
        }
        res = format_analysis_for_response(
            result=engine_result,
            case_id="999",
            patient_id="P1234",
            filename="test.png",
            image_bytes_len=254000
        )

        eff = next(p for p in res["predictions"] if p["disease"] == "Effusion")
        self.assertEqual(eff["raw_predicted_positive"], True)
        self.assertEqual(eff["calibrated_predicted_positive"], True)
        self.assertEqual(eff["raw_threshold"], 0.628)
        self.assertEqual(eff["calibrated_threshold"], 0.577)

        # OOD checks
        self.assertEqual(res["ood"]["lead_detector_name"], "Mahalanobis Distance")
        self.assertEqual(res["ood"]["triggers_count"], 0)
        self.assertEqual(res["ood"]["triggers_total"], 3)
        self.assertEqual(res["ood"]["score"], 636.927)
        self.assertEqual(res["ood"]["threshold"], 843.067)

        # Explainability spatial resolution check on heatmap
        heatmap_art = next(art for art in res["explanation"]["artifacts"] if art["name"] == "heatmap")
        self.assertTrue("7×7" in heatmap_art["caption"] or "7x7" in heatmap_art["caption"])
        self.assertIn("norm5", heatmap_art["caption"])

if __name__ == "__main__":
    unittest.main()
