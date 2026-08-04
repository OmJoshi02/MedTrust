from torchvision import transforms


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IMAGE_SIZE = 224

# DenseNet-121 will use ImageNet pretrained weights.
# Therefore inputs are normalized using ImageNet statistics.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------
# Training transforms
# ---------------------------------------------------------
#
# Training receives mild augmentation to improve
# generalization while avoiding aggressive transformations
# that could substantially distort chest X-ray anatomy.
#
# Original PNG files are NEVER modified.
# ---------------------------------------------------------

train_transform = transforms.Compose([

    # NIH images are loaded as grayscale ("L").
    # DenseNet-121 expects 3 input channels.
    transforms.Grayscale(num_output_channels=3),

    # Resize to DenseNet input resolution.
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Mild geometric augmentation.
    # Small rotations account for minor acquisition /
    # positioning differences.
    transforms.RandomRotation(
        degrees=7
    ),

    # Mild translation and scaling.
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
    ),

    # Mild intensity variation.
    transforms.ColorJitter(
        brightness=0.10,
        contrast=0.10,
    ),

    # PIL Image -> PyTorch tensor.
    # Pixel range becomes [0, 1].
    transforms.ToTensor(),

    # Normalize for ImageNet-pretrained DenseNet-121.
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ---------------------------------------------------------
# Validation / Test transforms
# ---------------------------------------------------------
#
# NO random augmentation is used during validation or test.
# This ensures deterministic and reproducible evaluation.
# ---------------------------------------------------------

eval_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=3),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# Separate names make later code clearer even though
# validation and test currently use identical transforms.

val_transform = eval_transform
test_transform = eval_transform