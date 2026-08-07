from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.dataset import CLASSES
from src.dataloader import create_dataloaders
from src.model import create_model


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
OUTPUT_DIR = Path("outputs/evaluation")

BATCH_SIZE = 16
NUM_WORKERS = 4


# ============================================================
# Load trained model
# ============================================================

def load_best_model(device):
    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    # Verify class order stored during training.
    if "classes" in checkpoint:
        checkpoint_classes = list(checkpoint["classes"])

        if checkpoint_classes != list(CLASSES):
            raise RuntimeError(
                "Checkpoint class order does not match "
                "src.dataset.CLASSES."
            )

        print("Checkpoint class order : VERIFIED")

    model = create_model(
        num_classes=len(CLASSES),
        pretrained=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("Checkpoint epoch       :", checkpoint.get("epoch", "unknown"))

    if "best_macro_auroc" in checkpoint:
        print(
            "Saved best Macro AUROC :",
            f"{float(checkpoint['best_macro_auroc']):.5f}",
        )

    return model, checkpoint


# ============================================================
# Validation inference
# ============================================================

def run_validation_inference(model, loader, device):
    all_targets = []
    all_logits = []
    all_probabilities = []
    all_filenames = []

    total_batches = len(loader)

    print("\nRunning validation inference...")
    print("-" * 78)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):

            images, targets, filenames = batch

            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            if not torch.isfinite(logits).all():
                raise RuntimeError(
                    f"Non-finite logits detected in batch {batch_idx}."
                )

            if not torch.isfinite(probabilities).all():
                raise RuntimeError(
                    f"Non-finite probabilities detected "
                    f"in batch {batch_idx}."
                )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_logits.append(
                logits.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_filenames.extend(filenames)

            if batch_idx % 250 == 0 or batch_idx == total_batches:
                print(
                    f"Batch {batch_idx:4d}/{total_batches}"
                )

    targets = np.concatenate(
        all_targets,
        axis=0,
    )

    logits = np.concatenate(
        all_logits,
        axis=0,
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    return (
        targets,
        logits,
        probabilities,
        all_filenames,
    )


# ============================================================
# AUROC
# ============================================================

def calculate_aurocs(targets, probabilities):
    results = []
    valid_aurocs = []

    print("\nVALIDATION AUROC BY DISEASE")
    print("-" * 78)

    for class_idx, disease in enumerate(CLASSES):

        y_true = targets[:, class_idx]
        y_score = probabilities[:, class_idx]

        positive_count = int(y_true.sum())
        negative_count = int(len(y_true) - positive_count)

        # AUROC requires both classes to be present.
        if positive_count == 0 or negative_count == 0:
            auroc = np.nan
        else:
            auroc = roc_auc_score(
                y_true,
                y_score,
            )

            valid_aurocs.append(auroc)

        results.append(
            {
                "disease": disease,
                "positives": positive_count,
                "negatives": negative_count,
                "auroc": float(auroc),
            }
        )

        print(
            f"{disease:<24}"
            f"{positive_count:>7,} pos   "
            f"AUROC = {auroc:.5f}"
        )

    macro_auroc = float(
        np.mean(valid_aurocs)
    )

    print("-" * 78)
    print(
        f"{'MACRO AUROC':<24}"
        f"{macro_auroc:.5f}"
    )

    return results, macro_auroc


# ============================================================
# Save validation predictions
# ============================================================

def save_predictions(
    filenames,
    targets,
    logits,
    probabilities,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "filename": filenames,
    }

    for idx, disease in enumerate(CLASSES):
        safe_name = disease.lower()

        data[f"target_{safe_name}"] = (
            targets[:, idx].astype(int)
        )

        data[f"logit_{safe_name}"] = (
            logits[:, idx]
        )

        data[f"prob_{safe_name}"] = (
            probabilities[:, idx]
        )

    df = pd.DataFrame(data)

    output_path = (
        OUTPUT_DIR /
        "validation_predictions.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nValidation predictions saved:",
        output_path,
    )


# ============================================================
# Save validation metrics
# ============================================================

def save_metrics(results, macro_auroc, checkpoint):
    metrics_df = pd.DataFrame(results)

    metrics_path = (
        OUTPUT_DIR /
        "validation_auroc.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    summary = {
        "split": "validation",
        "num_images": 16828,
        "num_classes": len(CLASSES),
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "saved_best_macro_auroc": (
            float(checkpoint["best_macro_auroc"])
            if "best_macro_auroc" in checkpoint
            else None
        ),
        "recomputed_macro_auroc": macro_auroc,
    }

    summary_path = (
        OUTPUT_DIR /
        "validation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "Validation AUROC saved     :",
        metrics_path,
    )

    print(
        "Validation summary saved   :",
        summary_path,
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 78)
    print("MEDTRUST — VALIDATION EVALUATION")
    print("=" * 78)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device                  :", device)

    if torch.cuda.is_available():
        print(
            "GPU                     :",
            torch.cuda.get_device_name(0),
        )

    print("Checkpoint              :", CHECKPOINT_PATH)
    print("Expected validation size: 16,828")
    print("Expected Macro AUROC     : ~0.82777")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    _, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    actual_val_size = len(
        val_loader.dataset
    )

    print(
        "Actual validation size  :",
        f"{actual_val_size:,}",
    )

    if actual_val_size != 16828:
        raise RuntimeError(
            "Unexpected validation dataset size. "
            f"Expected 16828, got {actual_val_size}."
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model, checkpoint = load_best_model(
        device
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    (
        targets,
        logits,
        probabilities,
        filenames,
    ) = run_validation_inference(
        model,
        val_loader,
        device,
    )

    print("\nINFERENCE CHECK")
    print("-" * 78)

    print(
        "Images processed        :",
        f"{len(filenames):,}",
    )

    print(
        "Target shape            :",
        targets.shape,
    )

    print(
        "Logit shape             :",
        logits.shape,
    )

    print(
        "Probability shape       :",
        probabilities.shape,
    )

    print(
        "Probability range       :",
        f"{probabilities.min():.6f}",
        "to",
        f"{probabilities.max():.6f}",
    )

    if targets.shape != (16828, len(CLASSES)):
        raise RuntimeError(
            f"Unexpected target shape: {targets.shape}"
        )

    if probabilities.shape != targets.shape:
        raise RuntimeError(
            "Probability and target shapes do not match."
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    results, macro_auroc = calculate_aurocs(
        targets,
        probabilities,
    )

    # --------------------------------------------------------
    # Compare against checkpoint
    # --------------------------------------------------------

    print("\nCHECKPOINT REPRODUCTION")
    print("-" * 78)

    saved_auroc = checkpoint.get(
        "best_macro_auroc"
    )

    if saved_auroc is not None:
        saved_auroc = float(saved_auroc)

        difference = abs(
            macro_auroc - saved_auroc
        )

        print(
            "Saved Macro AUROC       :",
            f"{saved_auroc:.6f}",
        )

        print(
            "Recomputed Macro AUROC  :",
            f"{macro_auroc:.6f}",
        )

        print(
            "Absolute difference     :",
            f"{difference:.8f}",
        )

        # Small floating-point differences are acceptable.
        if difference <= 1e-4:
            print(
                "Reproduction status    : PASSED"
            )
        else:
            print(
                "Reproduction status    : INVESTIGATE"
            )

            print(
                "\nSTOP: Do not perform final test evaluation "
                "until this discrepancy is understood."
            )

    # --------------------------------------------------------
    # Save artifacts
    # --------------------------------------------------------

    save_predictions(
        filenames,
        targets,
        logits,
        probabilities,
    )

    save_metrics(
        results,
        macro_auroc,
        checkpoint,
    )

    print("\n" + "=" * 78)
    print("VALIDATION EVALUATION COMPLETE")
    print("=" * 78)

    print(
        "\nNo test-set thresholds were selected."
    )

    print(
        "No final test evaluation was performed."
    )

    print(
        "Next step only after AUROC reproduction passes:"
    )

    print(
        "Validation-only disease threshold selection."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()