from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy.stats import spearmanr

from src.dataset import CLASSES
from src.explainability.gradcam import GradCAM
from src.transformer import test_transform


class ExplanationReliabilityAnalyzer:
    """
    Explanation Reliability Analyzer for MedTrust (Module 6).

    Evaluates whether Grad-CAM explanations behave reliably under input
    perturbations and tests explanation fidelity via region occlusion.

    Scientific Guarantees:
    - Reuses the existing GradCAM implementation (src.explainability.gradcam).
    - Preserves model state (eval mode, zero-grad, no weight/bias modification).
    - Reproducible perturbations via fixed random seed.
    - Transparent multi-metric evaluation (Pearson, Spearman, MAD, IoU, Fidelity).
    - Never reduces evidence to a single opaque score.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
        random_seed: int = 42,
        occlusion_fraction: float = 0.20,
        brightness_factor: float = 1.05,
        contrast_factor: float = 1.05,
        translation_pixels: int = 4,
        noise_std: float = 0.01,
        high_threshold: float = 0.90,
        moderate_threshold: float = 0.75,
    ):
        if not (0.0 < occlusion_fraction < 1.0):
            raise ValueError(
                f"occlusion_fraction must be in (0, 1), got {occlusion_fraction}"
            )
        if brightness_factor <= 0.0 or contrast_factor <= 0.0:
            raise ValueError("Brightness and contrast factors must be positive.")
        if translation_pixels < 0:
            raise ValueError("translation_pixels must be non-negative.")
        if noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if not (0.0 <= moderate_threshold <= high_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0.0 <= moderate_threshold <= high_threshold <= 1.0"
            )

        self.model = model
        self.target_layer = (
            target_layer
            if target_layer is not None
            else getattr(model.features, "norm5", None)
        )
        if self.target_layer is None:
            raise ValueError(
                "Target layer could not be determined. Please supply target_layer explicitly."
            )

        if device is not None:
            self.device = device
        else:
            self.device = next(model.parameters()).device

        self.random_seed = random_seed
        self.occlusion_fraction = occlusion_fraction
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.translation_pixels = translation_pixels
        self.noise_std = noise_std
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold

    # =========================================================
    # INPUT PREPARATION
    # =========================================================

    def _prepare_inputs(
        self,
        image_input: Union[str, Path, Image.Image, torch.Tensor],
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Normalize input into both PIL Image and preprocessed tensor [1, 3, 224, 224].
        """
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            pil_image = Image.open(path).convert("RGB")
            tensor = test_transform(pil_image).unsqueeze(0).to(self.device)
            return pil_image, tensor

        elif isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            tensor = test_transform(pil_image).unsqueeze(0).to(self.device)
            return pil_image, tensor

        elif isinstance(image_input, torch.Tensor):
            if image_input.ndim == 3:
                image_input = image_input.unsqueeze(0)
            if image_input.ndim != 4 or image_input.size(1) not in (1, 3):
                raise ValueError(
                    f"Expected tensor of shape [1, 3, H, W] or [3, H, W], got {tuple(image_input.shape)}"
                )
            if image_input.size(1) == 1:
                image_input = image_input.repeat(1, 3, 1, 1)

            tensor = image_input.to(self.device).float()

            # Create an approximate PIL representation for PIL-based transformations
            # Un-normalize using ImageNet stats: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
            denorm = (tensor * std + mean).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            pil_image = Image.fromarray((denorm * 255.0).astype(np.uint8))
            return pil_image, tensor

        else:
            raise TypeError(
                f"Unsupported image_input type: {type(image_input)}. Must be Path, str, PIL.Image, or torch.Tensor."
            )

    # =========================================================
    # REPRODUCIBLE PERTURBATIONS
    # =========================================================

    def _apply_brightness(self, image: Image.Image) -> torch.Tensor:
        perturbed = ImageEnhance.Brightness(image).enhance(self.brightness_factor)
        return test_transform(perturbed).unsqueeze(0).to(self.device)

    def _apply_contrast(self, image: Image.Image) -> torch.Tensor:
        perturbed = ImageEnhance.Contrast(image).enhance(self.contrast_factor)
        return test_transform(perturbed).unsqueeze(0).to(self.device)

    def _apply_translation(self, image: Image.Image) -> torch.Tensor:
        perturbed = image.transform(
            image.size,
            Image.AFFINE,
            (1, 0, self.translation_pixels, 0, 1, 0),
        )
        return test_transform(perturbed).unsqueeze(0).to(self.device)

    def _apply_gaussian_noise(
        self,
        base_tensor: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator(device=base_tensor.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            base_tensor.shape,
            generator=generator,
            device=base_tensor.device,
            dtype=base_tensor.dtype,
        ) * self.noise_std
        return base_tensor + noise

    # =========================================================
    # SPATIAL COMPARISON METRICS
    # =========================================================

    @staticmethod
    def compute_pearson_correlation(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
        """
        Pearson correlation coefficient between two 2D heatmaps.
        Handles flat/constant CAMs robustly.
        """
        a_flat = cam_a.flatten().astype(np.float64)
        b_flat = cam_b.flatten().astype(np.float64)

        std_a = np.std(a_flat)
        std_b = np.std(b_flat)

        if std_a < 1e-8 and std_b < 1e-8:
            return 1.0 if np.allclose(a_flat, b_flat, atol=1e-5) else 0.0
        if std_a < 1e-8 or std_b < 1e-8:
            return 0.0

        corr = np.corrcoef(a_flat, b_flat)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float(np.clip(corr, -1.0, 1.0))

    @staticmethod
    def compute_spearman_correlation(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
        """
        Spearman rank correlation coefficient between two 2D heatmaps.
        """
        a_flat = cam_a.flatten().astype(np.float64)
        b_flat = cam_b.flatten().astype(np.float64)

        if np.std(a_flat) < 1e-8 and np.std(b_flat) < 1e-8:
            return 1.0 if np.allclose(a_flat, b_flat, atol=1e-5) else 0.0
        if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
            return 0.0

        res = spearmanr(a_flat, b_flat)
        stat = getattr(res, "statistic", getattr(res, "correlation", 0.0))
        if np.isnan(stat):
            return 0.0
        return float(np.clip(stat, -1.0, 1.0))

    @staticmethod
    def compute_mean_absolute_difference(cam_a: np.ndarray, cam_b: np.ndarray) -> float:
        """
        Mean Absolute Difference (MAD / L1) between normalized CAMs in [0, 1].
        """
        return float(np.mean(np.abs(cam_a - cam_b)))

    @staticmethod
    def compute_region_iou(
        cam_a: np.ndarray,
        cam_b: np.ndarray,
        top_fraction: float = 0.20,
    ) -> float:
        """
        Intersection over Union (IoU) of the top K% attribution regions.
        """
        thresh_a = np.quantile(cam_a, 1.0 - top_fraction)
        thresh_b = np.quantile(cam_b, 1.0 - top_fraction)

        mask_a = cam_a >= thresh_a
        mask_b = cam_b >= thresh_b

        intersection = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()

        if union == 0:
            return 1.0
        return float(intersection / (union + 1e-8))

    # =========================================================
    # REGION FIDELITY (OCCLUSION TEST)
    # =========================================================

    def evaluate_fidelity(
        self,
        base_tensor: torch.Tensor,
        base_cam: np.ndarray,
        class_index: int,
    ) -> Dict[str, Union[float, int, bool]]:
        """
        Evaluate explanation fidelity by masking the highest-attribution region
        and measuring the resulting drop in model probability for that pathology.
        """
        # Threshold for top attribution pixels
        quantile_thresh = float(np.quantile(base_cam, 1.0 - self.occlusion_fraction))
        occlusion_mask_2d = base_cam >= quantile_thresh
        occluded_pixel_count = int(np.sum(occlusion_mask_2d))

        # Occlude normalized input tensor: set top attribution pixels to 0.0 (ImageNet mean)
        mask_tensor = torch.from_numpy(occlusion_mask_2d).to(
            device=base_tensor.device,
            dtype=torch.bool,
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        occluded_tensor = base_tensor.clone()
        occluded_tensor = occluded_tensor.masked_fill(mask_tensor, 0.0)

        self.model.eval()
        with torch.inference_mode():
            logits_orig = self.model(base_tensor)
            prob_orig = float(torch.sigmoid(logits_orig[0, class_index]).item())

            logits_occl = self.model(occluded_tensor)
            prob_occl = float(torch.sigmoid(logits_occl[0, class_index]).item())

        prob_drop = float(prob_orig - prob_occl)
        relative_drop = float(prob_drop / max(prob_orig, 1e-6))

        return {
            "fidelity_probability_before": prob_orig,
            "fidelity_probability_after": prob_occl,
            "fidelity_probability_drop": prob_drop,
            "fidelity_relative_drop": relative_drop,
            "occlusion_fraction": self.occlusion_fraction,
            "occluded_pixels": occluded_pixel_count,
            "total_pixels": int(base_cam.size),
            "fidelity_verified": bool(prob_drop > 0.0),
        }

    # =========================================================
    # CORE ANALYSIS
    # =========================================================

    def analyze(
        self,
        image_input: Union[str, Path, Image.Image, torch.Tensor],
        class_index: int,
    ) -> Dict:
        """
        Run complete explanation reliability and fidelity analysis for one pathology.

        Parameters
        ----------
        image_input:
            Path, PIL Image, or preprocessed PyTorch tensor [1, 3, H, W].
        class_index:
            Target pathology index (0-13).

        Returns
        -------
        Structured dictionary containing perturbation stability indicators,
        individual comparison metrics, and region fidelity results.
        """
        if not (0 <= class_index < len(CLASSES)):
            raise IndexError(
                f"class_index={class_index} is outside valid range 0-{len(CLASSES) - 1}."
            )

        pil_image, base_tensor = self._prepare_inputs(image_input)
        target_disease = CLASSES[class_index]

        # -----------------------------------------------------
        # 1. Baseline Grad-CAM (Reusing existing GradCAM class)
        # -----------------------------------------------------
        with GradCAM(model=self.model, target_layer=self.target_layer) as gradcam:
            baseline_cam_tensor, baseline_logits = gradcam.generate(
                input_tensor=base_tensor,
                class_index=class_index,
            )

        baseline_cam = baseline_cam_tensor.cpu().numpy()
        baseline_prob = float(torch.sigmoid(baseline_logits[0, class_index]).item())

        # -----------------------------------------------------
        # 2. Perturbation Definitions
        # -----------------------------------------------------
        perturbations_data = [
            (
                "brightness",
                self._apply_brightness(pil_image),
                {"factor": self.brightness_factor},
            ),
            (
                "contrast",
                self._apply_contrast(pil_image),
                {"factor": self.contrast_factor},
            ),
            (
                "translation",
                self._apply_translation(pil_image),
                {"pixels": self.translation_pixels},
            ),
            (
                "gaussian_noise",
                self._apply_gaussian_noise(base_tensor, seed=self.random_seed),
                {"std": self.noise_std, "seed": self.random_seed},
            ),
        ]

        # -----------------------------------------------------
        # 3. Evaluate Perturbations
        # -----------------------------------------------------
        perturbation_results = []
        pearson_scores = []
        spearman_scores = []
        mad_scores = []
        iou_scores = []

        with GradCAM(model=self.model, target_layer=self.target_layer) as gradcam:
            for name, pert_tensor, params in perturbations_data:
                pert_cam_tensor, pert_logits = gradcam.generate(
                    input_tensor=pert_tensor,
                    class_index=class_index,
                )
                pert_cam = pert_cam_tensor.cpu().numpy()
                pert_prob = float(torch.sigmoid(pert_logits[0, class_index]).item())

                p_corr = self.compute_pearson_correlation(baseline_cam, pert_cam)
                s_corr = self.compute_spearman_correlation(baseline_cam, pert_cam)
                mad = self.compute_mean_absolute_difference(baseline_cam, pert_cam)
                iou = self.compute_region_iou(
                    baseline_cam,
                    pert_cam,
                    top_fraction=self.occlusion_fraction,
                )

                pearson_scores.append(p_corr)
                spearman_scores.append(s_corr)
                mad_scores.append(mad)
                iou_scores.append(iou)

                # Descriptive category for this perturbation
                if p_corr >= self.high_threshold:
                    p_stability = "HIGH"
                elif p_corr >= self.moderate_threshold:
                    p_stability = "MODERATE"
                else:
                    p_stability = "LOW"

                perturbation_results.append(
                    {
                        "perturbation": name,
                        "parameters": params,
                        "pearson_correlation": p_corr,
                        "spearman_correlation": s_corr,
                        "mean_absolute_difference": mad,
                        "region_iou": iou,
                        "stability_band": p_stability,
                        "original_probability": baseline_prob,
                        "perturbed_probability": pert_prob,
                        "probability_change": float(pert_prob - baseline_prob),
                    }
                )

        mean_pearson = float(np.mean(pearson_scores))
        min_pearson = float(np.min(pearson_scores))
        mean_spearman = float(np.mean(spearman_scores))
        min_spearman = float(np.min(spearman_scores))
        mean_mad = float(np.mean(mad_scores))
        max_mad = float(np.max(mad_scores))
        mean_iou = float(np.mean(iou_scores))
        min_iou = float(np.min(iou_scores))

        if mean_pearson >= self.high_threshold:
            overall_stability = "HIGH"
        elif mean_pearson >= self.moderate_threshold:
            overall_stability = "MODERATE"
        else:
            overall_stability = "LOW"

        # -----------------------------------------------------
        # 4. Region Fidelity (Occlusion Test)
        # -----------------------------------------------------
        fidelity_results = self.evaluate_fidelity(
            base_tensor=base_tensor,
            base_cam=baseline_cam,
            class_index=class_index,
        )

        return {
            "target_disease": target_disease,
            "class_index": class_index,
            "original_probability": baseline_prob,
            "overall_stability": overall_stability,
            "mean_pearson_correlation": mean_pearson,
            "min_pearson_correlation": min_pearson,
            "mean_spearman_correlation": mean_spearman,
            "min_spearman_correlation": min_spearman,
            "mean_absolute_difference": mean_mad,
            "max_absolute_difference": max_mad,
            "mean_region_iou": mean_iou,
            "min_region_iou": min_iou,
            "thresholds": {
                "high": self.high_threshold,
                "moderate": self.moderate_threshold,
            },
            "perturbation_details": perturbation_results,
            "fidelity": fidelity_results,
            "fidelity_probability_before": fidelity_results["fidelity_probability_before"],
            "fidelity_probability_after": fidelity_results["fidelity_probability_after"],
            "fidelity_probability_drop": fidelity_results["fidelity_probability_drop"],
            "fidelity_verified": fidelity_results["fidelity_verified"],
            "scientific_interpretation": (
                "Explanation reliability quantifies Grad-CAM spatial stability under small, "
                "reproducible input perturbations and validates that masking high-attribution "
                "regions decreases model confidence. It does not establish ground-truth clinical diagnostic accuracy."
            ),
        }
