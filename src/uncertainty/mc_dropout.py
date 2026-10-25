"""
MC Dropout inference for the frozen MedTrust DenseNet-121.

Important design choice
-----------------------
The frozen MedTrust classifier is DenseNet-121 with a final Linear(1024, 14)
classifier and no Dropout layer. We do NOT modify the architecture or
checkpoint.

Instead, during each stochastic forward pass we:
1. run the frozen DenseNet feature extractor,
2. apply ReLU and adaptive average pooling exactly as DenseNet does,
3. apply dropout to the 1024-dimensional representation with training=True,
4. pass the dropped representation through the unchanged frozen classifier.

This preserves checkpoint compatibility while producing stochastic
predictions suitable for Monte Carlo Dropout uncertainty estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MCDropoutResult:
    """Container for stochastic MC-Dropout outputs."""

    logits: torch.Tensor
    probabilities: torch.Tensor

    @property
    def mean_probability(self) -> torch.Tensor:
        return self.probabilities.mean(dim=0)

    @property
    def probability_variance(self) -> torch.Tensor:
        # Population variance is used because these are Monte Carlo samples
        # describing predictive variability.
        return self.probabilities.var(dim=0, unbiased=False)

    @property
    def probability_std(self) -> torch.Tensor:
        return self.probability_variance.sqrt()


class MCDropoutPredictor:
    """
    Run Monte Carlo Dropout inference without changing the trained model.

    Parameters
    ----------
    model:
        Frozen DenseNet-121 model created by src.model.create_model().
    passes:
        Number of stochastic forward passes.
    dropout_rate:
        Probability of dropping each feature dimension.
    device:
        Torch device used for inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        passes: int = 30,
        dropout_rate: float = 0.20,
        device: torch.device | str = "cpu",
    ) -> None:
        if passes < 2:
            raise ValueError("passes must be at least 2.")

        if not 0.0 < dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in (0, 1).")

        self.model = model
        self.passes = int(passes)
        self.dropout_rate = float(dropout_rate)
        self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _stochastic_forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform one stochastic forward pass.

        The feature extractor remains in evaluation mode. Only the explicit
        feature dropout is stochastic.
        """
        features = self.model.features(image_tensor)
        features = F.relu(features, inplace=False)

        pooled = F.adaptive_avg_pool2d(features, (1, 1))
        representation = torch.flatten(pooled, start_dim=1)

        dropped = F.dropout(
            representation,
            p=self.dropout_rate,
            training=True,
        )

        logits = self.model.classifier(dropped)

        return logits

    @torch.inference_mode()
    def predict(self, image_tensor: torch.Tensor) -> MCDropoutResult:
        """
        Run the requested number of stochastic forward passes.

        Returns
        -------
        MCDropoutResult
            logits shape: [passes, batch, classes]
            probabilities shape: [passes, batch, classes]
        """
        if image_tensor.ndim != 4:
            raise ValueError(
                "image_tensor must have shape [batch, channels, height, width]."
            )

        image_tensor = image_tensor.to(self.device)

        logits_samples = []
        probability_samples = []

        for _ in range(self.passes):
            logits = self._stochastic_forward(image_tensor)
            probabilities = torch.sigmoid(logits)

            logits_samples.append(logits)
            probability_samples.append(probabilities)

        logits_stack = torch.stack(logits_samples, dim=0)
        probability_stack = torch.stack(probability_samples, dim=0)

        return MCDropoutResult(
            logits=logits_stack,
            probabilities=probability_stack,
        )
