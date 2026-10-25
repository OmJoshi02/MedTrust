import numpy as np
import torch
import torch.nn as nn

from src.uncertainty.mc_dropout import MCDropoutPredictor
from src.uncertainty.uncertainty_analyzer import UncertaintyAnalyzer
from src.uncertainty.uncertainty_metrics import (
    binary_entropy,
    predictive_variance,
    variation_ratio,
)


class TinyDenseNetLike(nn.Module):
    """
    Small model with the same interface used by MCDropoutPredictor.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=1),
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


def test_predictive_variance():
    samples = np.array(
        [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
        ]
    )

    variance = predictive_variance(samples)

    assert np.allclose(
        variance,
        [0.0266666667, 0.0266666667],
    )


def test_entropy_is_maximum_near_half():
    entropy = binary_entropy(np.array([0.5]))

    assert np.isclose(
        entropy[0],
        np.log(2.0),
    )


def test_variation_ratio_zero_when_all_agree():
    samples = np.array(
        [
            [0.8, 0.2],
            [0.9, 0.1],
            [0.7, 0.3],
        ]
    )

    vr = variation_ratio(samples)

    assert np.allclose(vr, [0.0, 0.0])


def test_variation_ratio_increases_with_disagreement():
    samples = np.array(
        [
            [0.8],
            [0.2],
            [0.8],
            [0.2],
        ]
    )

    vr = variation_ratio(samples)

    assert np.isclose(vr[0], 0.5)


def test_mc_dropout_shape_and_stochasticity():
    torch.manual_seed(42)

    model = TinyDenseNetLike()

    predictor = MCDropoutPredictor(
        model=model,
        passes=8,
        dropout_rate=0.5,
        device="cpu",
    )

    image = torch.ones(1, 3, 8, 8)

    result = predictor.predict(image)

    assert result.probabilities.shape == (8, 1, 2)
    assert result.logits.shape == (8, 1, 2)

    # With dropout enabled, repeated stochastic passes should normally
    # differ for this non-degenerate feature representation.
    assert not torch.allclose(
        result.probabilities[0],
        result.probabilities[1],
    )


def test_uncertainty_analyzer():
    samples = np.array(
        [
            [0.05, 0.90],
            [0.15, 0.80],
            [0.35, 0.70],
            [0.85, 0.60],
        ]
    )

    analyzer = UncertaintyAnalyzer(
        ["Atelectasis", "Cardiomegaly"]
    )

    result = analyzer.analyze(samples)

    assert result["passes"] == 4
    assert len(result["diseases"]) == 2
    assert result["most_uncertain_disease"] == "Atelectasis"
    assert (
        result["diseases"][0].variance
        > result["diseases"][1].variance
    )


import unittest


class TestUncertainty(unittest.TestCase):
    def test_predictive_variance(self):
        test_predictive_variance()

    def test_entropy_is_maximum_near_half(self):
        test_entropy_is_maximum_near_half()

    def test_variation_ratio_zero_when_all_agree(self):
        test_variation_ratio_zero_when_all_agree()

    def test_variation_ratio_increases_with_disagreement(self):
        test_variation_ratio_increases_with_disagreement()

    def test_mc_dropout_shape_and_stochasticity(self):
        test_mc_dropout_shape_and_stochasticity()

    def test_uncertainty_analyzer(self):
        test_uncertainty_analyzer()


if __name__ == "__main__":
    unittest.main()
