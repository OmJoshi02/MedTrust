import torch
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


NUM_CLASSES = 14


def create_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
):
    """
    Create DenseNet-121 for NIH ChestX-ray14 multi-label classification.

    Input:
        [B, 3, 224, 224]

    Output:
        [B, 14] raw logits

    IMPORTANT:
        No sigmoid is included inside the model.
        BCEWithLogitsLoss will be used during training.
    """

    if pretrained:
        weights = DenseNet121_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = densenet121(weights=weights)

    # DenseNet-121's original classifier:
    # Linear(1024 -> 1000)
    #
    # Replace it with:
    # Linear(1024 -> 14)
    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        num_classes,
    )

    return model


if __name__ == "__main__":
    model = create_model()

    print("=" * 65)
    print("MEDTRUST DENSENET-121")
    print("=" * 65)

    print("Classifier:")
    print(model.classifier)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print("=" * 65)