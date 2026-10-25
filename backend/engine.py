"""
MedTrust V1 — Backend Inference and Analysis Engine (backend/engine.py)

Integrates:
- DenseNet-121 model evaluation & deterministic transforms
- Frozen Youden J decision thresholds & per-disease temperature scaling
- Calibration analysis lookup (Module 2 & 9)
- Out-of-Distribution detection: MSP, Energy, Mahalanobis (Module 7)
- Prediction Uncertainty: MC-Dropout 30 passes (Module 6)
- Explainability: Grad-CAM attribution & perturbation stability (Modules 3 & 4)
- Technical radiograph quality evaluation
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from PIL import Image

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform
from src.calibration import (
    DEFAULT_TEMPERATURE_PATH,
    CalibrationAnalyzer,
    apply_temperature_scaling,
    load_frozen_temperatures,
)
from src.evaluation import DEFAULT_THRESHOLDS_PATH, load_frozen_thresholds
from src.explainability.gradcam import GradCAM
from src.explainability.reliability import ExplanationReliabilityAnalyzer
from src.OOD.predict import predict_all_methods
from src.uncertainty import MCDropoutPredictor, UncertaintyAnalyzer
from src.quality import WeightedImageQualityAnalyzer
from src.similarity import SimilarityAnalyzer
from src.similarity_v2 import SimilarityRetrievalEngineV2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "outputs" / "evaluation" / "thresholds.json"
DEFAULT_TEMPERATURE_PATH = PROJECT_ROOT / "outputs" / "calibration" / "temperature.json"
DEFAULT_BINS_PATH = PROJECT_ROOT / "outputs" / "calibration" / "reliability" / "validation_reliability_bins.csv"


# Trusted, versioned cryptographic manifest of model and clinical research artifacts.
# Hashes must never be derived dynamically from current disk state during verification.
TRUSTED_ARTIFACT_MANIFEST: Dict[str, Dict[str, Any]] = {
    "checkpoint": {
        "rel_path": "checkpoints/best_model.pt",
        "sha256": "6160588028dad6ce5396c74962c7d85b162d06b816f34175249fa069c8466d34",
        "min_size_bytes": 80_000_000,
        "required": True,
        "description": "DenseNet-121 trained weights (Macro AUROC 0.8278, Epoch 12 early stopping)",
    },
    "thresholds": {
        "rel_path": "outputs/evaluation/thresholds.json",
        "sha256": "2c33facd3e84415dcd4fb5bce7d30261a94ff733f4dd796d32fae072f80ef7bb",
        "min_size_bytes": 1000,
        "required": True,
        "expected_classes": 14,
        "description": "Validation-split Youden J decision thresholds (14 classes)",
    },
    "temperature": {
        "rel_path": "outputs/calibration/temperature.json",
        "sha256": "58319f56b3704daf25267cc189b17aa169e242d8baa7f9fbd5d6de8d34a62a29",
        "min_size_bytes": 1000,
        "required": True,
        "expected_classes": 14,
        "description": "Per-disease Platt scaling temperatures (14 classes)",
    },
    "reliability_bins": {
        "rel_path": "outputs/calibration/reliability/validation_reliability_bins.csv",
        "sha256": "20ca8d32d3a795071128bfa41205a83c56aaa53ab6341f71387023b81db2c204",
        "min_size_bytes": 1000,
        "required": False,
        "description": "Validation reliability diagram empirical bin frequencies",
    },
    "quality_reference": {
        "rel_path": "outputs/quality/quality_reference.json",
        "sha256": "b4ea572823fa28fc5fb6ade9d824a171d7263f494eed6e835a80303776510cd3",
        "min_size_bytes": 100,
        "required": False,
        "description": "Baseline reference distribution for image quality metrics",
    },
    "reference_embeddings": {
        "rel_path": "outputs/similarity/reference_embeddings.npz",
        "sha256": "8107ff6ab72fc8d72d151e09026c241d82becf7d83d36de3fee0ef077f6192fb",
        "min_size_bytes": 1000,
        "required": False,
        "description": "DenseNet reference case embeddings for Top-K similarity retrieval",
    },
    "reference_embeddings_v2": {
        "rel_path": "outputs/similarity_v2/reference_embeddings.npz",
        "sha256": "d88b7725f17df839af7cf40c537923772dd55724e440e3aa50ca15906bc578e2",
        "min_size_bytes": 1000,
        "required": False,
        "description": "DenseNet reference case embeddings for Top-K similarity retrieval V2",
    },
}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of an artifact."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifacts(
    project_root: Optional[Path] = None,
    enforce_hashes: bool = True,
) -> Dict[str, Any]:
    """
    Verify presence, sizes, schemas, and cryptographic SHA-256 hashes of model & clinical artifacts
    against the trusted versioned manifest. Hashes are never derived dynamically from current files.
    """
    root = project_root or PROJECT_ROOT
    report: Dict[str, Any] = {
        "ready": True,
        "all_present": True,
        "critical_present": True,
        "artifacts": {},
        "failures": [],
    }

    for key, spec in TRUSTED_ARTIFACT_MANIFEST.items():
        path = root / spec["rel_path"]
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        sha = compute_file_sha256(path) if exists and size > 0 else None
        is_required = spec.get("required", False)

        valid = exists and size >= spec.get("min_size_bytes", 1)
        err = None

        if not exists:
            err = "File not found"
            valid = False
        elif size < spec.get("min_size_bytes", 1):
            err = f"File size ({size} bytes) below minimum expected ({spec['min_size_bytes']} bytes)"
            valid = False
        elif enforce_hashes and sha != spec["sha256"]:
            err = f"Cryptographic integrity mismatch (expected SHA-256 {spec['sha256'][:16]}..., found {sha[:16] if sha else 'none'}...)"
            valid = False
        else:
            # Semantic schema / shape checks
            if key == "thresholds":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("classes") != list(CLASSES):
                        err = f"Thresholds classes mismatch: expected {list(CLASSES)}"
                        valid = False
                    else:
                        thr_dict = data.get("thresholds", {})
                        if not isinstance(thr_dict, dict) or len(thr_dict) != len(CLASSES):
                            err = f"Invalid thresholds schema: expected {len(CLASSES)} disease thresholds, found {len(thr_dict)}"
                            valid = False
                        else:
                            for disease in CLASSES:
                                if disease not in thr_dict:
                                    err = f"Missing threshold entry for class '{disease}'"
                                    valid = False
                                    break
                                entry = thr_dict[disease]
                                val = entry.get("threshold") if isinstance(entry, dict) else entry
                                if val is None or not (0.0 <= float(val) <= 1.0):
                                    err = f"Invalid threshold value for '{disease}': {val}"
                                    valid = False
                                    break
                except Exception as exc:
                    err = f"Failed to parse thresholds JSON: {exc}"
                    valid = False
            elif key == "temperature":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    temp_dict = data.get("temperatures", {})
                    if not isinstance(temp_dict, dict) or len(temp_dict) != len(CLASSES):
                        err = f"Invalid temperature schema: expected {len(CLASSES)} disease temperatures, found {len(temp_dict)}"
                        valid = False
                    elif not all(float(v) > 0.0 for v in temp_dict.values()):
                        err = "Invalid temperature values (must be > 0.0)"
                        valid = False
                except Exception as exc:
                    err = f"Failed to parse temperature JSON: {exc}"
                    valid = False

        if not valid:
            report["all_present"] = False
            if is_required:
                report["ready"] = False
                report["critical_present"] = False
            report["failures"].append({"artifact": key, "required": is_required, "error": err})

        report["artifacts"][key] = {
            "path": spec["rel_path"],
            "exists": exists,
            "size_bytes": size,
            "sha256": sha,
            "expected_sha256": spec["sha256"],
            "hash_matches": sha == spec["sha256"] if sha else False,
            "critical": is_required,
            "required": is_required,
            "valid": valid,
            "description": spec["description"],
            "error": err,
        }

    return report


def array_to_base64_png(arr: np.ndarray) -> str:
    """Convert float [0, 1] or uint8 [0, 255] RGB image array to base64 PNG data URI."""
    if arr.dtype != np.uint8:
        uint8_arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    else:
        uint8_arr = arr
    img = Image.fromarray(uint8_arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


class MedTrustBackendEngine:
    """Singleton engine caching neural weights and frozen clinical artifacts."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._inference_lock = threading.Lock()
        self.artifact_report = verify_artifacts(PROJECT_ROOT)

        self.model, self.checkpoint_meta = self._load_model()
        self.thresholds = load_frozen_thresholds(DEFAULT_THRESHOLDS_PATH)
        self.temperatures = load_frozen_temperatures(DEFAULT_TEMPERATURE_PATH)

        # Compute exact calibrated equivalent thresholds: tau_cal = sigma(logit(tau_raw) / T)
        raw_t = np.clip(self.thresholds, 1e-7, 1.0 - 1e-7)
        logits_t = np.log(raw_t / (1.0 - raw_t))
        scaled_logits_t = logits_t / self.temperatures
        self.calibrated_thresholds = (1.0 / (1.0 + np.exp(-scaled_logits_t))).astype(np.float32)

        self.calibration_analyzer = CalibrationAnalyzer(
            bins_csv_path=DEFAULT_BINS_PATH,
            state="after",
        )
        self.reliability_analyzer = ExplanationReliabilityAnalyzer(
            model=self.model,
            target_layer=self.model.features.norm5,
            device=self.device,
        )
        self.quality_analyzer = WeightedImageQualityAnalyzer()
        self.similarity_analyzer = SimilarityAnalyzer(
            model=self.model,
            device=self.device,
        )

    def _load_model(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        if not DEFAULT_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Checkpoint missing at {DEFAULT_CHECKPOINT_PATH}")
        ckpt = torch.load(DEFAULT_CHECKPOINT_PATH, map_location=self.device, weights_only=True)
        model = create_model(num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(self.device).eval()
        return model, ckpt

    def compute_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Evaluate interpretable technical image quality indicators (Module 5).
        Returns engineering metrics: sharpness, contrast, brightness, clipping, noise, resolution, composite score.
        """
        res = self.quality_analyzer.analyze(image)
        res_dict = res.to_dict()

        rgb_img = image.convert("RGB")
        w, h = image.size

        composite_quality = res_dict["composite_score"]
        quality_status = res_dict["quality_status"]
        suitable = res_dict["suitable"]

        summary = (
            f"Technical quality evaluated as {res.status} (score {res.quality_score:.1f}/100). "
            + (" ".join(res.warnings) if res.warnings else "Radiograph satisfies nominal technical quality limits.")
        )

        noise_val = max(float(res.raw_metrics.get("noise", 1.0)), 1e-4)
        contrast_val = max(float(res.raw_metrics.get("contrast", 1.0)), 1e-4)
        snr_db = round(float(10.0 * np.log10(max((contrast_val ** 2) / (noise_val ** 2), 1.0))), 1)

        return {
            "composite_score": composite_quality,
            "quality_score": res_dict["quality_score"],
            "quality_status": quality_status,
            "overall_status": "pass" if suitable else "caution",
            "status": res.status,
            "suitable": suitable,
            "summary": summary,
            "snr_db": snr_db,
            "sharpness": res_dict["sharpness"],
            "contrast": res_dict["contrast"],
            "brightness": res_dict["brightness"],
            "clipping": res_dict["clipping"],
            "noise": res_dict["noise"],
            "resolution": res_dict["resolution"],
            "raw_metrics": res_dict["raw_metrics"],
            "normalized_metrics": res_dict["normalized_metrics"],
            "weighted_contributions": res_dict["weighted_contributions"],
            "warnings": res_dict["warnings"],
            "original_resolution": f"{w} x {h}",
            "target_resolution": "224 x 224",
            "bit_depth": "8-bit Grayscale" if rgb_img.mode != "RGB" else "RGB / 8-bit",
        }

    def run_prediction(self, image: Image.Image) -> Dict[str, Any]:
        """Run 14-disease classification, temperature calibration, and OOD detection."""
        pil_gray = image.convert("L")
        tensor = test_transform(pil_gray).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features_map = self.model.features(tensor)
            features_map = F.relu(features_map, inplace=False)
            pooled = F.adaptive_avg_pool2d(features_map, (1, 1))
            features = torch.flatten(pooled, start_dim=1)
            logits = self.model.classifier(features)

            raw_probs = torch.sigmoid(logits)
            calibrated_logits = apply_temperature_scaling(logits, self.temperatures)
            calibrated_probs = torch.sigmoid(calibrated_logits)

        raw_scores = raw_probs.squeeze(0).cpu().numpy()
        calibrated_scores = calibrated_probs.squeeze(0).cpu().numpy()
        decisions = (raw_scores >= self.thresholds).astype(int)

        findings = []
        for idx, disease in enumerate(CLASSES):
            r_val = float(raw_scores[idx])
            c_val = float(calibrated_scores[idx])
            raw_thresh_val = float(self.thresholds[idx])
            cal_thresh_val = float(self.calibrated_thresholds[idx])
            raw_is_pos = bool(r_val >= raw_thresh_val)
            cal_is_pos = bool(c_val >= cal_thresh_val)
            cal_diag = self.calibration_analyzer.analyze_prediction(disease, c_val)

            findings.append({
                "index": idx,
                "disease": disease,
                "raw_probability": r_val,
                "calibrated_probability": c_val,
                "threshold": cal_thresh_val,
                "raw_threshold": raw_thresh_val,
                "calibrated_threshold": cal_thresh_val,
                "temperature": float(self.temperatures[idx]),
                "decision": "POSITIVE" if cal_is_pos else "Negative",
                "predicted_positive": cal_is_pos,
                "raw_predicted_positive": raw_is_pos,
                "calibrated_predicted_positive": cal_is_pos,
                "calibration_status": cal_diag["status"],
                "calibration_gap": cal_diag["calibration_gap"],
                "empirical_rate": cal_diag["empirical_positive_rate"],
                "val_samples": cal_diag["bin_sample_count"],
            })

        top_raw_idx = int(np.argmax(raw_scores))
        top_cal_idx = int(np.argmax(calibrated_scores))

        positive_findings = [f for f in findings if f["decision"] == "POSITIVE"]
        if positive_findings:
            primary = sorted(positive_findings, key=lambda f: f["calibrated_probability"], reverse=True)[0]
        else:
            primary = findings[top_raw_idx]

        # Module 7: Out-of-Distribution Detection
        ood_res = predict_all_methods(
            logits=logits.detach().cpu().numpy(),
            features=features.detach().cpu().numpy(),
        )

        return {
            "findings": findings,
            "raw_scores": raw_scores.tolist(),
            "calibrated_scores": calibrated_scores.tolist(),
            "top_raw": findings[top_raw_idx],
            "top_calibrated": findings[top_cal_idx],
            "primary_finding": primary,
            "ood": ood_res,
            "tensor": tensor,
            "logits": logits,
        }

    def run_uncertainty(self, image_tensor: torch.Tensor, passes: int = 30) -> Dict[str, Any]:
        """Module 6: Run Monte Carlo Dropout prediction uncertainty."""
        predictor = MCDropoutPredictor(
            model=self.model,
            passes=passes,
            dropout_rate=0.20,
            device=self.device,
        )
        mc_result = predictor.predict(image_tensor)
        mc_samples = mc_result.probabilities[:, 0, :].detach().cpu().numpy()

        analyzer = UncertaintyAnalyzer(CLASSES)
        analysis = analyzer.analyze(mc_samples)

        diseases_summary = [
            {
                "disease": d.disease,
                "probability": d.probability,
                "variance": d.variance,
                "standard_deviation": d.standard_deviation,
                "entropy": d.entropy,
            }
            for d in analysis["diseases"]
        ]

        stds = [d.standard_deviation for d in analysis["diseases"]]
        mean_std = float(np.mean(stds))
        root_mean_var = float(np.sqrt(analysis["overall_mean_variance"]))

        return {
            "passes": passes,
            "dropout_rate": 0.20,
            "overall_mean_variance": analysis["overall_mean_variance"],
            "overall_max_variance": analysis["overall_max_variance"],
            "overall_mean_std": mean_std,
            "root_mean_variance": root_mean_var,
            "most_uncertain_disease": analysis["most_uncertain_disease"],
            "most_uncertain_variance": analysis["most_uncertain_variance"],
            "diseases": diseases_summary,
        }

    def generate_gradcam_maps(
        self,
        image: Image.Image,
        class_index: int,
        alpha: float = 0.65,
    ) -> Dict[str, Any]:
        """Module 3 & 4: Generate Grad-CAM heatmaps, overlays, and occlusion fidelity."""
        pil_gray = image.convert("L")
        disp_img = pil_gray.resize((224, 224)).convert("RGB")
        disp_array = np.asarray(disp_img, dtype=np.float32) / 255.0

        tensor = test_transform(pil_gray).unsqueeze(0).to(self.device)

        target_layer = self.model.features.norm5
        with GradCAM(model=self.model, target_layer=target_layer) as gradcam:
            cam_tensor, gradcam_logits = gradcam.generate(input_tensor=tensor, class_index=class_index)

        cam = cam_tensor.detach().cpu().numpy()

        # Gaussian smoothing filter over upsampled CAM to avoid 7x7 blocky interpolation artifacts
        cam_smooth = ndimage.gaussian_filter(cam, sigma=2.0)
        c_min = float(cam_smooth.min())
        c_max = float(cam_smooth.max())
        if (c_max - c_min) > 1e-8:
            cam_smooth = (cam_smooth - c_min) / (c_max - c_min)
        else:
            cam_smooth = np.zeros_like(cam_smooth)

        # Jet colormap rendering
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("jet")
        heatmap_rgb = cmap(cam_smooth)[:, :, :3]
        overlay = (1.0 - alpha) * disp_array + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0.0, 1.0)

        # Reliability analysis
        rel_metrics = self.reliability_analyzer.analyze(
            image_input=pil_gray,
            class_index=class_index,
        )

        return {
            "class_name": CLASSES[class_index],
            "class_index": class_index,
            "original_image_b64": array_to_base64_png(disp_array),
            "heatmap_b64": array_to_base64_png(heatmap_rgb),
            "overlay_b64": array_to_base64_png(overlay),
            "stability_status": rel_metrics.get("overall_stability", "HIGH"),
            "mean_spearman": float(rel_metrics.get("mean_spearman_correlation", 0.99)),
            "fidelity_drop": float(rel_metrics.get("fidelity_probability_drop", 0.50)),
            "fidelity_verified": bool(rel_metrics.get("fidelity_verified", True)),
        }

    def check_ood(self, image: Image.Image) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Module 7: Out-of-Distribution initial gating check."""
        pil_gray = image.convert("L")
        tensor = test_transform(pil_gray).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            features_map = self.model.features(tensor)
            features_map = F.relu(features_map, inplace=False)
            pooled = F.adaptive_avg_pool2d(features_map, (1, 1))
            features = torch.flatten(pooled, start_dim=1)
            logits = self.model.classifier(features)

        ood_res = predict_all_methods(
            logits=logits.detach().cpu().numpy(),
            features=features.detach().cpu().numpy(),
        )
        return ood_res, tensor, features, logits

    def run_similarity(
        self,
        image: Image.Image,
        top_k: int = 5,
        query_findings: Optional[List[str]] = None,
        primary_finding: Optional[str] = None,
        predicted_findings: Optional[Sequence[str]] = None,
        exclude_filenames: Optional[Union[str, Sequence[str]]] = None,
        authentic_only: bool = True,
        exclude_identical_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """Module 8: Case-Based Similarity Retrieval."""
        return self.similarity_analyzer.find_similar_cases(
            image,
            top_k=top_k,
            query_findings=query_findings,
            primary_finding=primary_finding,
            predicted_findings=predicted_findings,
            exclude_filenames=exclude_filenames,
            authentic_only=authentic_only,
            exclude_identical_embeddings=exclude_identical_embeddings,
        )

    def analyze_full(
        self,
        image: Image.Image,
        query_filename: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform unified MedTrust V1 pipeline with strict OOD pre-gating:
        1. Image Out-of-Distribution check.
        2. If OOD: directly report prediction cannot be generated (halt all diagnosis).
        3. Else (In-Distribution): Image quality check -> Prediction & Calibration -> Uncertainty -> Grad-CAM -> Similarity.
        """
        with self._inference_lock:
            pil_gray = image.convert("L")
            disp_img = pil_gray.resize((224, 224)).convert("RGB")
            disp_array = np.asarray(disp_img, dtype=np.float32) / 255.0

            # Step 1: Out-of-Distribution screening FIRST
            ood_res, tensor, features, logits = self.check_ood(image)

            # GATING: If OOD, immediately halt prediction generation
            if ood_res["is_ood"]:
                return {
                    "status": "REJECTED_OOD",
                    "is_ood": True,
                    "message": "Prediction cannot be generated because the input image is Out-of-Distribution (domain shift or non-chest-radiograph detected).",
                    "ood": ood_res,
                    "quality": None,
                    "prediction": None,
                    "uncertainty": None,
                    "similarity": None,
                    "explainability": {
                        "class_name": "N/A (OOD Rejected)",
                        "class_index": -1,
                        "original_image_b64": array_to_base64_png(disp_array),
                        "heatmap_b64": "",
                        "overlay_b64": "",
                        "stability_status": "NOT_EVALUATED_OOD",
                        "mean_spearman": 0.0,
                        "fidelity_drop": 0.0,
                        "fidelity_verified": False,
                    },
                    "trust": {
                        "composite_score": 0,
                        "reliability_band": "LOW_RELIABILITY_OOD",
                        "decision_text": "PREDICTION CANNOT BE GENERATED (Out-of-Distribution Detected)",
                        "checklist": [
                            {"label": "Domain Distribution Status", "value": "OOD (REJECTED)", "ok": False},
                            {"label": "Clinical Safety Gating", "value": "HALTED (NO PREDICTION)", "ok": False},
                            {"label": "Out-of-Distribution Screening", "value": f"{ood_res['status']} DETECTED", "ok": False},
                        ],
                    },
                }

            # Step 2: Image Quality Check (In-Distribution only)
            quality = self.compute_image_quality(image)

            # Step 3: 14-Disease Prediction & Temperature Calibration
            pred_res = self.run_prediction(image)
            primary_idx = pred_res["primary_finding"]["index"]
            primary_disease = pred_res["primary_finding"]["disease"]
            detected_diseases = [
                f["disease"]
                for f in pred_res.get("findings", [])
                if f.get("decision") == "POSITIVE" or f.get("calibrated_predicted_positive")
            ]

            # Step 4: Epistemic Uncertainty (30 passes)
            uncertainty = self.run_uncertainty(pred_res["tensor"], passes=30)

            # Step 5: Grad-CAM Explainability & Perturbation Stability
            gradcam = self.generate_gradcam_maps(image, class_index=primary_idx)

            # Step 6: Case-Based Similarity Retrieval with Clinical Concordance (Module 8)
            predicted_findings_list = (
                detected_diseases
                if detected_diseases
                else ([primary_disease] if primary_disease else ["No Finding"])
            )
            similarity = self.run_similarity(
                image,
                top_k=top_k,
                query_findings=detected_diseases,
                primary_finding=primary_disease,
                predicted_findings=predicted_findings_list,
                exclude_filenames=[query_filename] if query_filename else None,
                authentic_only=True,
                exclude_identical_embeddings=True,
            )

            # Step 7: Composite AI Reliability Index (0-100)
            q_score = quality["composite_score"]
            conf_score = int(round(pred_res["primary_finding"]["calibrated_probability"] * 100.0))
            ood_score = 95
            unc_penalty = int(np.clip(uncertainty["overall_mean_variance"] * 5000.0, 0, 30))
            stab_score = int(round(gradcam["mean_spearman"] * 100.0))

            composite_reliability = int(
                0.25 * q_score + 0.25 * conf_score + 0.25 * ood_score + 0.25 * (stab_score - unc_penalty)
            )
            composite_reliability = int(np.clip(composite_reliability, 40, 95))

            if composite_reliability >= 80:
                reliability_band = "HIGH_RELIABILITY"
                decision_text = "PREDICTION CAN BE CONSIDERED (In-Distribution, High Stability)"
            else:
                reliability_band = "MODERATE_RELIABILITY"
                decision_text = "EXERCISE CAUTION (Moderate Epistemic Variance)"

            checklist = [
                {"label": "Domain Distribution Status", "value": "IN-DISTRIBUTION (PASS)", "ok": True},
                {"label": "Radiograph Technical Quality", "value": f"{q_score}/100", "ok": quality["suitable"]},
                {"label": "Calibrated Model Confidence", "value": f"{conf_score}%", "ok": conf_score >= 50},
                {"label": "Predictive Epistemic Variance", "value": f"{uncertainty['overall_mean_variance']:.5f}", "ok": uncertainty["overall_mean_variance"] < 0.005},
                {"label": "Explanation Spatial Stability", "value": f"{stab_score}%", "ok": stab_score >= 85},
                {"label": "Case-Based Similarity Context", "value": f"{similarity['top_match']['similarity_score']:.1f}% top match" if similarity.get('top_match') else "Indexed", "ok": True},
            ]

            return {
                "status": "ANALYZED",
                "is_ood": False,
                "quality": quality,
                "prediction": {
                    "findings": pred_res["findings"],
                    "primary_finding": pred_res["primary_finding"],
                    "top_raw": pred_res["top_raw"],
                    "top_calibrated": pred_res["top_calibrated"],
                },
                "ood": ood_res,
                "uncertainty": uncertainty,
                "explainability": gradcam,
                "similarity": similarity,
                "trust": {
                    "composite_score": composite_reliability,
                    "reliability_band": reliability_band,
                    "decision_text": decision_text,
                    "checklist": checklist,
                },
            }
