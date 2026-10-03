"""
MedTrust V1 — Master Diagnostic Inference Pipeline (src/inference/pipeline.py)

Unifies the core functional modules with strict separation of concerns:
- Evaluation     : src.dataset, frozen Youden J thresholds
- Calibration    : src.calibration.calibration_analyzer (CalibrationAnalyzer) & frozen temperatures
- Explainability : src.explainability.gradcam (GradCAM) & src.explainability.reliability (ExplanationReliabilityAnalyzer)
- Preprocessing  : src.transformer (Standardized radiographic transformations)
- Architecture   : src.model (Frozen DenseNet-121 classifier)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.dataset import CLASSES
from src.model import create_model
from src.transformer import test_transform
from src.explainability.gradcam import GradCAM
from src.calibration.calibration_analyzer import CalibrationAnalyzer
from src.explainability.reliability import ExplanationReliabilityAnalyzer

DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
DEFAULT_THRESHOLDS_PATH = Path("outputs/evaluation/thresholds.json")
DEFAULT_TEMPERATURE_PATH = Path("outputs/calibration/temperature.json")
DEFAULT_GRADCAM_OUTPUT_DIR = Path("outputs/explainability/gradcam")
DEFAULT_INFERENCE_OUTPUT_DIR = Path("outputs/inference")


def load_frozen_thresholds(path: Optional[Path] = None) -> np.ndarray:
    p = path or DEFAULT_THRESHOLDS_PATH
    if not p.exists():
        raise FileNotFoundError(f"Thresholds artifact not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("classes") != list(CLASSES):
        raise RuntimeError("Threshold class order does not match canonical CLASSES.")
    return np.array(
        [data["thresholds"][disease]["threshold"] for disease in CLASSES],
        dtype=np.float32,
    )


def load_frozen_temperatures(path: Optional[Path] = None) -> np.ndarray:
    p = path or DEFAULT_TEMPERATURE_PATH
    if not p.exists():
        raise FileNotFoundError(f"Calibration artifact not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("classes") != list(CLASSES):
        raise RuntimeError("Calibration class order does not match canonical CLASSES.")
    if data.get("calibration_split") != "validation":
        raise RuntimeError("Calibration artifact was not fitted on validation data.")
    return np.array(
        [float(data["temperatures"][disease]) for disease in CLASSES],
        dtype=np.float32,
    )


def save_gradcam_visualization(
    display_image: np.ndarray,
    cam: np.ndarray,
    class_name: str,
    raw_probability: float,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(display_image, cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    # Heatmap
    heatmap = axes[1].imshow(cam, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"Grad-CAM\n{class_name}")
    axes[1].axis("off")
    fig.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(display_image)
    axes[2].imshow(cam, cmap="jet", alpha=0.40, vmin=0.0, vmax=1.0)
    axes[2].set_title(f"{class_name}\nRaw probability: {raw_probability:.4f}")
    axes[2].axis("off")

    fig.suptitle("MedTrust — Disease-Specific Grad-CAM", fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


class MedTrustInferencePipeline:
    """
    Master trust-aware inference engine unifying prediction, calibration,
    explainability, and explanation reliability.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        thresholds_path: Optional[Union[str, Path]] = None,
        temperatures_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self.thresholds_path = Path(thresholds_path) if thresholds_path else DEFAULT_THRESHOLDS_PATH
        self.temperatures_path = Path(temperatures_path) if temperatures_path else DEFAULT_TEMPERATURE_PATH
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_INFERENCE_OUTPUT_DIR
        self.gradcam_dir = DEFAULT_GRADCAM_OUTPUT_DIR

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradcam_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load parameters and model
        self.thresholds = load_frozen_thresholds(self.thresholds_path)
        self.temperatures = load_frozen_temperatures(self.temperatures_path)
        self.model, self.checkpoint_meta = self._load_model()

        # Initialize analyzers
        self.calibration_analyzer = CalibrationAnalyzer(state="after")
        self.reliability_analyzer = ExplanationReliabilityAnalyzer(
            model=self.model,
            device=self.device,
        )

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if list(checkpoint["classes"]) != list(CLASSES):
            raise RuntimeError("Checkpoint classes do not match CLASSES.")
        model = create_model(num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        return model, checkpoint

    def run(
        self,
        image_input: Union[str, Path, Image.Image],
        target_disease: Optional[str] = None,
        skip_gradcam: bool = False,
        skip_reliability: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the end-to-end MedTrust diagnostic inference pipeline on one chest X-ray.
        """
        # Resolve image input
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            pil_image = Image.open(img_path).convert("RGB")
            stem = img_path.stem
        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            stem = "xray_sample"
            img_path = Path(f"{stem}.png")
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        # 1. Preprocessing
        image_tensor = test_transform(pil_image).unsqueeze(0).to(self.device)

        # 2. Model Inference & Temperature Calibration
        with torch.inference_mode():
            logits = self.model(image_tensor)
            raw_probs = torch.sigmoid(logits)
            t_tensor = torch.tensor(self.temperatures, dtype=logits.dtype, device=self.device).unsqueeze(0)
            calibrated_probs = torch.sigmoid(logits / t_tensor)

        raw_scores = raw_probs.squeeze(0).cpu().numpy()
        calibrated_scores = calibrated_probs.squeeze(0).cpu().numpy()

        # Decision rule: raw probability vs frozen Youden J threshold
        decisions = (raw_scores >= self.thresholds).astype(int)

        # 3. Determine top predictions and findings
        raw_top_idx = int(np.argmax(raw_scores))
        calibrated_top_idx = int(np.argmax(calibrated_scores))

        positive_indices = np.where(decisions == 1)[0]
        if len(positive_indices) > 0:
            ordered_indices = sorted(positive_indices, key=lambda i: calibrated_scores[i], reverse=True)
        else:
            ordered_indices = [raw_top_idx]

        # 4. Calibration Diagnostics
        findings = []
        for idx, d_name in enumerate(CLASSES):
            c_score = float(calibrated_scores[idx])
            diag = self.calibration_analyzer.analyze_prediction(d_name, c_score)
            findings.append({
                "disease": d_name,
                "raw_probability": float(raw_scores[idx]),
                "calibrated_probability": c_score,
                "threshold": float(self.thresholds[idx]),
                "decision": "POSITIVE" if decisions[idx] == 1 else "Negative",
                "temperature": float(self.temperatures[idx]),
                "calibration_bin": diag["bin_range"],
                "empirical_positive_rate": diag["empirical_positive_rate"],
                "calibration_gap": diag["calibration_gap"],
                "calibration_status": diag["status"],
            })

        # 5. Resolve target disease for Explainability
        if target_disease:
            if target_disease not in CLASSES:
                raise ValueError(f"Unknown target_disease '{target_disease}'. Must be one of {CLASSES}")
            explain_idx = CLASSES.index(target_disease)
        else:
            explain_idx = ordered_indices[0]

        explain_disease_name = CLASSES[explain_idx]

        # 6. Grad-CAM Explainability
        gradcam_path = None
        if not skip_gradcam:
            target_layer = self.model.features.norm5
            with GradCAM(model=self.model, target_layer=target_layer) as gradcam:
                cam_tensor, _ = gradcam.generate(input_tensor=image_tensor, class_index=explain_idx)
            cam = cam_tensor.cpu().numpy()

            disp_img = np.asarray(pil_image.resize((224, 224))).astype(np.float32) / 255.0
            safe_name = explain_disease_name.lower().replace(" ", "_")
            out_file = self.gradcam_dir / f"{stem}__{safe_name}__gradcam.png"
            gradcam_path = save_gradcam_visualization(
                display_image=disp_img,
                cam=cam,
                class_name=explain_disease_name,
                raw_probability=float(raw_scores[explain_idx]),
                output_path=out_file,
            )

        # 7. Explanation Reliability & Occlusion Fidelity
        reliability_res = None
        if not skip_reliability:
            reliability_res = self.reliability_analyzer.analyze(
                image_input=pil_image,
                class_index=explain_idx,
            )

        # 8. Compile Unified Structured Record
        structured_result = {
            "image": str(img_path),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint_meta.get("epoch", None),
            "model_architecture": "DenseNet-121",
            "threshold_source": "validation-only Youden J (FROZEN)",
            "calibration_source": "validation-only Temperature Scaling (FROZEN)",
            "top_prediction_raw": {
                "disease": CLASSES[raw_top_idx],
                "probability": float(raw_scores[raw_top_idx]),
            },
            "top_prediction_calibrated": {
                "disease": CLASSES[calibrated_top_idx],
                "probability": float(calibrated_scores[calibrated_top_idx]),
            },
            "findings": findings,
            "explanation": {
                "target_disease": explain_disease_name,
                "target_layer": "model.features.norm5",
                "gradcam_artifact": str(gradcam_path) if gradcam_path else None,
            },
            "explanation_reliability": reliability_res,
            "clinical_disclaimer": "This research model output is not a clinical diagnosis.",
        }

        # Save machine-readable JSON artifact
        json_path = self.output_dir / f"{stem}__trust_result.json"

        def _json_serializable(val):
            if isinstance(val, (np.integer, np.int64, np.int32)):
                return int(val)
            if isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, Path):
                return str(val)
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            return str(val)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_result, f, indent=4, default=_json_serializable)

        structured_result["json_artifact"] = str(json_path)
        return structured_result
