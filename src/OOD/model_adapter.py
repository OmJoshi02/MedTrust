"""
Adapter around the existing MedTrust DenseNet-121 model.

This file intentionally reuses:
    src.model.create_model
    src.dataset.CLASSES

No duplicate DenseNet architecture is created here.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from src.dataset import CLASSES
from src.model import create_model

from .config import (
    CHECKPOINT_PATH,
    FEATURE_DIM,
    NUM_CLASSES,
)
from .common import (
    get_device,
    require_file,
    validate_class_order,
)


class MedTrustModelAdapter:
    """
    Loads the frozen MedTrust DenseNet-121 checkpoint and exposes:

        logits
        probabilities
        1024-dimensional feature representation
    """

    def __init__(
        self,
        checkpoint_path=CHECKPOINT_PATH,
        device=None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.device = (
            device
            if device is not None
            else get_device()
        )

        self.model = None
        self.checkpoint = None

        self._load_model()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self):

        require_file(
            self.checkpoint_path,
            "MedTrust DenseNet-121 checkpoint",
        )

        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        validate_class_order(
            self.checkpoint.get("classes"),
            CLASSES,
        )

        if "model_state_dict" not in self.checkpoint:

            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        self.model = create_model(
            num_classes=NUM_CLASSES,
            pretrained=False,
        )

        self.model.load_state_dict(
            self.checkpoint[
                "model_state_dict"
            ]
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        self._validate_classifier()

    # ========================================================
    # CLASSIFIER VALIDATION
    # ========================================================

    def _validate_classifier(self):

        classifier = getattr(
            self.model,
            "classifier",
            None,
        )

        if classifier is None:

            raise RuntimeError(
                "DenseNet classifier layer not found."
            )

        if not hasattr(
            classifier,
            "in_features",
        ):

            raise RuntimeError(
                "Classifier does not expose in_features."
            )

        if classifier.in_features != FEATURE_DIM:

            raise RuntimeError(
                "Unexpected DenseNet feature dimension: "
                f"{classifier.in_features}; "
                f"expected {FEATURE_DIM}."
            )

        if classifier.out_features != NUM_CLASSES:

            raise RuntimeError(
                "Unexpected classifier output dimension: "
                f"{classifier.out_features}; "
                f"expected {NUM_CLASSES}."
            )

    # ========================================================
    # DEVICE
    # ========================================================

    def to(self, device):

        self.device = torch.device(
            device
        )

        self.model = self.model.to(
            self.device
        )

        return self

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features_and_logits(
        self,
        images,
    ):
        """
        Extract:

            features: [batch, 1024]
            logits:   [batch, 14]
        """

        if images.ndim != 4:

            raise ValueError(
                "Input images must have shape "
                "[batch, channels, height, width]."
            )

        images = images.to(
            self.device
        )

        with torch.inference_mode():

            # DenseNet feature extractor
            features_map = self.model.features(
                images
            )

            # DenseNet uses ReLU after feature blocks
            features_map = F.relu(
                features_map,
                inplace=False,
            )

            # Global average pooling
            pooled = F.adaptive_avg_pool2d(
                features_map,
                output_size=(1, 1),
            )

            features = torch.flatten(
                pooled,
                start_dim=1,
            )

            logits = self.model.classifier(
                features
            )

        if features.shape[1] != FEATURE_DIM:

            raise RuntimeError(
                "Unexpected feature dimension: "
                f"{features.shape[1]}; "
                f"expected {FEATURE_DIM}."
            )

        if logits.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                "Unexpected logits dimension: "
                f"{logits.shape[1]}; "
                f"expected {NUM_CLASSES}."
            )

        return (
            features,
            logits,
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        images,
    ):
        """
        Return 1024-D features, logits and sigmoid probabilities.
        """

        features, logits = (
            self.extract_features_and_logits(
                images
            )
        )

        probabilities = torch.sigmoid(
            logits
        )

        return {
            "features": features,
            "logits": logits,
            "probabilities": probabilities,
        }

    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

    @property
    def epoch(self):

        return self.checkpoint.get(
            "epoch",
            None,
        )

    @property
    def best_macro_auroc(self):

        return self.checkpoint.get(
            "best_macro_auroc",
            None,
        )

    @property
    def classes(self):

        return list(CLASSES)

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self):

        return {
            "checkpoint": str(
                self.checkpoint_path
            ),
            "device": str(
                self.device
            ),
            "architecture": "DenseNet-121",
            "feature_dimension": FEATURE_DIM,
            "num_classes": NUM_CLASSES,
            "classes": list(CLASSES),
            "checkpoint_epoch": self.epoch,
            "best_validation_macro_auroc": (
                self.best_macro_auroc
            ),
        }