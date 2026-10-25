"""
MedTrust OOD Detector — Core Standalone Inference Engine (src/OOD/detector.py)

Evaluates Out-of-Distribution (OOD) status for chest radiographs.
Produces conservative, non-dogmatic categories:
    - in-distribution-like
    - borderline/uncertain
    - out-of-distribution-like
in compliance with MedTrust AGENTS.md Rule 17.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from PIL import Image

from .config import (
    ARTIFACT_OUTPUT_DIR,
    ENERGY_TEMPERATURE,
    THRESHOLD_OUTPUT_PATH,
)
from .scores import (
    energy_score,
    msp_score,
    mahalanobis_score,
)
from .common import get_device


# Default provisional thresholds if no validation artifact exists yet
DEFAULT_THRESHOLDS = {
    "energy": 0.0,         # Energy-based score threshold
    "msp": 0.50,          # MSP score (1 - max_sigmoid) threshold
    "mahalanobis": 1200.0 # Mahalanobis distance threshold
}


@dataclass
class OODResult:
    """Structured result returned by OODDetector."""
    method: str
    score: float
    threshold: float
    status: str          # "in-distribution-like", "borderline/uncertain", "out-of-distribution-like"
    is_ood: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def format_report(self) -> str:
        """Format a clean, readable ASCII terminal report."""
        sep = "=" * 80
        thin_sep = "-" * 80
        lines = [
            sep,
            "MEDTRUST - OUT-OF-DISTRIBUTION (OOD) DETECTION REPORT",
            sep,
            f"Method              : {self.method.upper()}",
            f"OOD Score           : {self.score:10.4f}",
            f"Decision Threshold  : {self.threshold:10.4f}",
            f"Conservative Status : {self.status.upper()}",
            f"Flagged OOD         : {'YES (Caution)' if self.is_ood else 'NO'}",
            thin_sep,
            f"Summary             : {self.summary}",
            thin_sep,
            "NOTE: OOD detection is an engineering reliability indicator for distribution shift.",
            "It does NOT evaluate clinical disease presence or replace medical judgment.",
            sep,
        ]
        return "\n".join(lines)


class OODDetector:
    """
    Standalone Out-of-Distribution detector for MedTrust.
    
    Can evaluate:
        1. Pre-computed logits (zero-overhead, no model re-run)
        2. PyTorch image tensor [1, 3, 224, 224]
        3. Local radiograph image file path
    """

    def __init__(
        self,
        method: str = "energy",
        temperature: float = ENERGY_TEMPERATURE,
        threshold: Optional[float] = None,
        margin: float = 0.15,
        model: Optional[torch.nn.Module] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Parameters
        ----------
        method : str
            Scoring method: 'energy' (default), 'msp', or 'mahalanobis'.
        temperature : float
            Temperature parameter for energy score.
        threshold : Optional[float]
            Decision threshold. If None, loaded from validation artifact or defaults.
        margin : float
            Relative uncertainty boundary margin for borderline/uncertain categorization.
        model : Optional[torch.nn.Module]
            Existing loaded DenseNet model (reused to prevent redundant memory allocation).
        device : Optional[device]
            PyTorch compute device.
        """
        method_lower = method.lower()
        if method_lower not in {"energy", "msp", "mahalanobis"}:
            raise ValueError(
                f"Unsupported OOD method: {method}. Choose from: 'energy', 'msp', 'mahalanobis'."
            )

        self.method = method_lower
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.device = get_device(device)
        self.model = model

        # Load threshold
        if threshold is not None:
            self.threshold = float(threshold)
        else:
            self.threshold = self._load_threshold(self.method)

        # Mahalanobis statistics cache
        self._mahalanobis_stats: Optional[Dict[str, np.ndarray]] = None

    def _load_threshold(self, method: str) -> float:
        """Attempt to load validation threshold from artifact; fallback to default."""
        if THRESHOLD_OUTPUT_PATH.is_file():
            try:
                import json
                with open(THRESHOLD_OUTPUT_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "thresholds" in data and method in data["thresholds"]:
                    return float(data["thresholds"][method]["threshold"])
            except Exception:
                pass
        return DEFAULT_THRESHOLDS.get(method, 0.0)

    def _get_mahalanobis_statistics(self) -> Dict[str, np.ndarray]:
        """Load frozen Mahalanobis feature statistics if needed."""
        if self._mahalanobis_stats is not None:
            return self._mahalanobis_stats

        stats_path = ARTIFACT_OUTPUT_DIR / "mahalanobis_statistics.npz"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Mahalanobis statistics artifact not found at {stats_path}. "
                "Fit Mahalanobis statistics on training features first, or use method='energy'/'msp'."
            )

        data = np.load(stats_path, allow_pickle=False)
        self._mahalanobis_stats = {
            "mean": np.asarray(data["mean"], dtype=np.float64),
            "safe_variance": np.asarray(data["safe_variance"], dtype=np.float64),
        }
        return self._mahalanobis_stats

    def compute_score(
        self,
        logits: Optional[Union[np.ndarray, torch.Tensor]] = None,
        features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> float:
        """Calculate OOD score for given logits/features."""
        if self.method in {"energy", "msp"}:
            if logits is None:
                raise ValueError(f"Method '{self.method}' requires logits.")
            if isinstance(logits, torch.Tensor):
                logits_np = logits.detach().cpu().numpy()
            else:
                logits_np = np.asarray(logits, dtype=np.float64)

            if logits_np.ndim == 1:
                logits_np = logits_np.reshape(1, -1)

            if self.method == "energy":
                scores = energy_score(logits_np, temperature=self.temperature)
            else:
                scores = msp_score(logits_np)
            return float(scores[0])

        elif self.method == "mahalanobis":
            if features is None:
                raise ValueError("Method 'mahalanobis' requires feature vectors.")
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features, dtype=np.float64)

            if features_np.ndim == 1:
                features_np = features_np.reshape(1, -1)

            stats = self._get_mahalanobis_statistics()
            scores = mahalanobis_score(
                features=features_np,
                mean=stats["mean"],
                safe_variance=stats["safe_variance"],
            )
            return float(scores[0])

        raise ValueError(f"Unknown method: {self.method}")

    def categorize(self, score: float) -> tuple[str, bool, str]:
        """
        Classify score into conservative tri-state categories per AGENTS.md Rule 17:
            - in-distribution-like
            - borderline/uncertain
            - out-of-distribution-like
        """
        # Threshold width band based on margin
        band = abs(self.threshold) * self.margin if abs(self.threshold) > 1e-4 else 0.5 * self.margin
        lower_bound = self.threshold - band
        upper_bound = self.threshold + band

        if score < lower_bound:
            status = "in-distribution-like"
            is_ood = False
            summary = "Sample aligns with the in-distribution reference feature distribution."
        elif score <= upper_bound:
            status = "borderline/uncertain"
            is_ood = False
            summary = "Sample falls within the uncertainty margin near the decision boundary."
        else:
            status = "out-of-distribution-like"
            is_ood = True
            summary = "Sample exhibits significant distribution shift from training data. Interpret predictions with caution."

        return status, is_ood, summary

    def detect(
        self,
        image_path: Optional[Union[str, Path]] = None,
        image_tensor: Optional[torch.Tensor] = None,
        logits: Optional[Union[np.ndarray, torch.Tensor]] = None,
        features: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> OODResult:
        """
        Perform OOD detection on input.
        
        Accepts pre-computed logits/features directly, OR runs model forward pass
        if image_tensor or image_path is provided.
        """
        # If logits/features not provided, run forward pass
        if logits is None and features is None:
            if image_tensor is None:
                if image_path is None:
                    raise ValueError("Must provide either logits, image_tensor, or image_path.")
                
                # Load image from path
                from src.transformer import eval_transform
                img = Image.open(image_path).convert("L")
                image_tensor = eval_transform(img).unsqueeze(0)

            if self.model is None:
                raise ValueError("Model is required to extract logits from image tensor.")

            image_tensor = image_tensor.to(self.device)
            self.model.eval()
            with torch.inference_mode():
                # Extract features if DenseNet architecture
                if hasattr(self.model, "features"):
                    f_map = self.model.features(image_tensor)
                    f_map = torch.relu(f_map)
                    pooled = torch.nn.functional.adaptive_avg_pool2d(f_map, (1, 1))
                    extracted_features = torch.flatten(pooled, 1)
                    if hasattr(self.model, "classifier"):
                        extracted_logits = self.model.classifier(extracted_features)
                    else:
                        extracted_logits = self.model(image_tensor)
                else:
                    extracted_logits = self.model(image_tensor)
                    extracted_features = None

            logits = extracted_logits
            features = extracted_features

        # Compute score and conservative categorization
        score = self.compute_score(logits=logits, features=features)
        status, is_ood, summary = self.categorize(score)

        return OODResult(
            method=self.method,
            score=score,
            threshold=self.threshold,
            status=status,
            is_ood=is_ood,
            summary=summary,
            details={
                "margin": self.margin,
                "temperature": self.temperature,
            },
        )
