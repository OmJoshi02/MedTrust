"""
MedTrust V1 — DenseNet-121 Feature Extractor for Similarity Analysis
(src/similarity/feature_extractor.py)

Module 8: Similarity Analysis.
Extracts 1024-dimensional visual feature representations from radiographs
using the trained DenseNet-121 convolutional backbone.

Feature Representation Flow:
    Input Image (Chest X-ray)
        ↓
    test_transform (Grayscale -> 3-ch, Resize 224x224, ImageNet Normalization)
        ↓
    model.features (DenseNet-121 backbone, 1024x7x7)
        ↓
    ReLU activation
        ↓
    AdaptiveAvgPool2d(1, 1) -> (1024, 1, 1)
        ↓
    Flatten -> 1024-dimensional vector
        ↓
    Finite value & dimension validation (D = 1024, no NaN/Inf)
        ↓
    L2 Normalization -> Unit norm (||v||_2 = 1.0)
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.transformer import test_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file for provenance verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class DenseNetSimilarityExtractor:
    """
    Extracts 1024-dimensional L2-normalized feature embeddings from chest radiographs
    using the verified trained DenseNet-121 convolutional backbone.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        allow_fallback: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.checkpoint_sha256: Optional[str] = None
        self.feature_dim = 1024

        if model is not None:
            # Reuse pre-loaded model (e.g. from MedTrustBackendEngine)
            self.model = model.to(self.device).eval()
            self.features = self.model.features
            self.checkpoint_identity = "Pre-loaded Model Instance"
        else:
            if self.checkpoint_path.exists():
                from src.dataset import CLASSES
                from src.model import create_model

                m = create_model(num_classes=len(CLASSES), pretrained=False)
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                m.load_state_dict(ckpt["model_state_dict"])
                self.model = m.to(self.device).eval()
                self.features = self.model.features
                try:
                    self.checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)
                except Exception:
                    self.checkpoint_sha256 = None
                self.checkpoint_identity = f"checkpoints/best_model.pt (Epoch {ckpt.get('epoch', 'N/A')})"
            elif allow_fallback:
                # Standalone fallback only when explicitly permitted
                from torchvision.models import DenseNet121_Weights, densenet121

                self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
                self.model = self.model.to(self.device).eval()
                self.features = self.model.features
                self.checkpoint_identity = "Torchvision ImageNet Pretrained (Fallback)"
            else:
                raise FileNotFoundError(
                    f"Trained MedTrust DenseNet-121 checkpoint not found at: {self.checkpoint_path}. "
                    "Ensure checkpoints/best_model.pt is present before generating reference embeddings."
                )

        self.features.eval()

    def get_provenance_manifest(self) -> Dict[str, Any]:
        """Return provenance manifest describing model, checkpoint, and preprocessing."""
        return {
            "model_architecture": "DenseNet-121",
            "feature_layer": "model.features (norm5 -> relu -> adaptive_avg_pool2d(1, 1))",
            "feature_dimension": self.feature_dim,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "preprocessing": "Grayscale -> 3-ch, Resize(224, 224), ImageNet Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            "normalization": "L2 unit-norm (norm = 1.0)",
        }

    def preprocess_image(self, image_input: Union[str, Path, bytes, Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Safely open and preprocess image into a PyTorch tensor of shape (1, 3, 224, 224).
        Follows MedTrust frozen evaluation preprocessing contract.
        """
        if isinstance(image_input, torch.Tensor):
            t = image_input
            if t.ndim == 3:
                t = t.unsqueeze(0)
            return t.to(self.device)

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
            img = Image.open(p)
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Ensure single-channel grayscale or RGB conversion
        if img.mode != "L" and img.mode != "RGB":
            img = img.convert("RGB")
        elif img.mode == "RGB":
            # Grayscale transform expects L mode for NIH contract
            img = img.convert("L")

        tensor = test_transform(img)
        return tensor.unsqueeze(0).to(self.device)

    def extract_features(
        self,
        image_input: Union[str, Path, bytes, Image.Image, torch.Tensor],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Extract 1024-dimensional feature vector from an input radiograph.

        Args:
            image_input: Radiograph file path, raw bytes, PIL Image, or PyTorch tensor.
            normalize: If True (default), L2-normalizes the vector to unit length (norm = 1.0).

        Returns:
            1D float32 NumPy array of shape (1024,).
        """
        tensor = self.preprocess_image(image_input)

        with torch.inference_mode():
            feat_map = self.features(tensor)  # (1, 1024, 7, 7)
            out = F.relu(feat_map, inplace=False)
            pooled = F.adaptive_avg_pool2d(out, (1, 1))  # (1, 1024, 1, 1)
            vector = torch.flatten(pooled, start_dim=1).squeeze(0).cpu().numpy().astype(np.float32)

        # Integrity checks
        if vector.shape != (self.feature_dim,):
            raise ValueError(f"Extracted feature dimension mismatch: expected ({self.feature_dim},), got {vector.shape}")

        if not np.all(np.isfinite(vector)):
            raise ValueError("Extracted feature vector contains non-finite values (NaN or Inf).")

        if normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector = vector / norm
            else:
                vector = np.zeros_like(vector)

        return vector

    def extract_batch(
        self,
        image_inputs: Sequence[Union[str, Path, bytes, Image.Image, torch.Tensor]],
        normalize: bool = True,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Extract 1024-dimensional feature vectors in batches for high-throughput processing.

        Args:
            image_inputs: Sequence of radiograph inputs (paths, PIL images, or tensors).
            normalize: If True (default), L2-normalizes each vector to unit length.
            batch_size: Batch size for forward passes (default: 64).

        Returns:
            2D float32 NumPy array of shape (N, 1024).
        """
        if len(image_inputs) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        results: List[np.ndarray] = []
        n_total = len(image_inputs)

        for i in range(0, n_total, batch_size):
            chunk = image_inputs[i : i + batch_size]
            tensors_list: List[torch.Tensor] = []
            for item in chunk:
                t = self.preprocess_image(item)  # (1, 3, 224, 224)
                tensors_list.append(t)

            batch_tensor = torch.cat(tensors_list, dim=0).to(self.device)  # (B, 3, 224, 224)

            with torch.inference_mode():
                feat_map = self.features(batch_tensor)  # (B, 1024, 7, 7)
                out = F.relu(feat_map, inplace=False)
                pooled = F.adaptive_avg_pool2d(out, (1, 1))  # (B, 1024, 1, 1)
                vectors = torch.flatten(pooled, start_dim=1)  # (B, 1024)

                if normalize:
                    norms = torch.norm(vectors, p=2, dim=1, keepdim=True).clamp_min(1e-12)
                    vectors = vectors / norms

                batch_arr = vectors.cpu().numpy().astype(np.float32)

            if batch_arr.shape[1] != self.feature_dim:
                raise ValueError(
                    f"Extracted feature dimension mismatch: expected ({self.feature_dim},), got {batch_arr.shape[1]}"
                )

            if not np.all(np.isfinite(batch_arr)):
                raise ValueError("Extracted feature vector contains non-finite values (NaN or Inf).")

            results.append(batch_arr)

        return np.vstack(results)

