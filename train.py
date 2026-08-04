from pathlib import Path
import csv
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.dataloader import create_dataloaders
from src.dataset import CLASSES
from src.loss import create_loss_function
from src.model import create_model


# =========================================================
# CONFIGURATION
# =========================================================

SEED = 42

BATCH_SIZE = 16
NUM_WORKERS = 4

MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 5

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "last_checkpoint.pt"
HISTORY_PATH = LOG_DIR / "training_history.csv"


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
):
    model.train()

    running_loss = 0.0
    total_samples = 0

    start_time = time.time()

    for batch_idx, (images, targets, _) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        # Progress every 500 batches
        if (batch_idx + 1) % 500 == 0:

            elapsed = time.time() - start_time

            print(
                f"  Epoch {epoch:02d}"
                f" | Batch {batch_idx + 1:5d}/{len(loader)}"
                f" | Loss {loss.item():.4f}"
                f" | Time {elapsed / 60:.1f} min"
            )

    epoch_loss = running_loss / total_samples

    return epoch_loss


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probabilities = []

    for images, targets, _ in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda"),
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        probabilities = torch.sigmoid(logits)

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(
            targets.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.float().cpu().numpy()
        )

    val_loss = running_loss / total_samples

    targets_np = np.concatenate(
        all_targets,
        axis=0,
    )

    probabilities_np = np.concatenate(
        all_probabilities,
        axis=0,
    )

    # -----------------------------------------------------
    # Per-class AUROC
    # -----------------------------------------------------

    class_aurocs = []

    for class_idx in range(len(CLASSES)):

        y_true = targets_np[:, class_idx]
        y_score = probabilities_np[:, class_idx]

        # roc_auc_score requires both positive and negative
        # samples to be present.
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y_true,
                y_score,
            )

        class_aurocs.append(auc)

    macro_auroc = float(
        np.nanmean(class_aurocs)
    )

    return (
        val_loss,
        macro_auroc,
        class_aurocs,
    )


# =========================================================
# CHECKPOINT
# =========================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scaler,
    scheduler,
    best_macro_auroc,
    epochs_without_improvement,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_macro_auroc": best_macro_auroc,
        "epochs_without_improvement": epochs_without_improvement,
        "classes": CLASSES,
        "seed": SEED,
    }

    torch.save(
        checkpoint,
        path,
    )


# =========================================================
# HISTORY
# =========================================================

def initialize_history():

    if HISTORY_PATH.exists():
        return

    with open(
        HISTORY_PATH,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_macro_auroc",
            "learning_rate",
            *[
                f"auc_{disease}"
                for disease in CLASSES
            ],
        ])


def append_history(
    epoch,
    train_loss,
    val_loss,
    macro_auroc,
    learning_rate,
    class_aurocs,
):
    with open(
        HISTORY_PATH,
        "a",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            macro_auroc,
            learning_rate,
            *class_aurocs,
        ])


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    initialize_history()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print("MEDTRUST — DENSENET-121 TRAINING")
    print("=" * 78)

    print("Device        :", device)

    if device.type == "cuda":
        print(
            "GPU           :",
            torch.cuda.get_device_name(0)
        )

    print("Batch size    :", BATCH_SIZE)
    print("Max epochs    :", MAX_EPOCHS)
    print("Learning rate :", LEARNING_RATE)
    print("Weight decay  :", WEIGHT_DECAY)
    print("Early stopping:", EARLY_STOPPING_PATIENCE)

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print()
    print("Train samples :", len(train_loader.dataset))
    print("Val samples   :", len(val_loader.dataset))

    # IMPORTANT:
    # Test loader is intentionally NOT used here.

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = create_model(
        num_classes=len(CLASSES),
        pretrained=True,
    ).to(device)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion, pos_weights = create_loss_function(
        device=device
    )

    print()
    print(
        "Positive weights:",
        [round(x, 3) for x in pos_weights.cpu().tolist()]
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Reduce LR when validation AUROC stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    # -----------------------------------------------------
    # Resume support
    # -----------------------------------------------------

    start_epoch = 1
    best_macro_auroc = -np.inf
    epochs_without_improvement = 0

    if LAST_CHECKPOINT_PATH.exists():

        print()
        print(
            "Existing checkpoint found:",
            LAST_CHECKPOINT_PATH
        )

        checkpoint = torch.load(
            LAST_CHECKPOINT_PATH,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        best_macro_auroc = checkpoint[
            "best_macro_auroc"
        ]

        epochs_without_improvement = checkpoint[
            "epochs_without_improvement"
        ]

        print(
            f"Resuming from epoch {start_epoch}"
        )

        print(
            f"Best validation macro AUROC: "
            f"{best_macro_auroc:.4f}"
        )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING START")
    print("=" * 78)

    for epoch in range(
        start_epoch,
        MAX_EPOCHS + 1,
    ):

        epoch_start = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch {epoch}/{MAX_EPOCHS}"
            f" | LR = {current_lr:.2e}"
        )

        print("-" * 78)

        # TRAIN
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        # VALIDATION
        (
            val_loss,
            macro_auroc,
            class_aurocs,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_minutes = (
            time.time() - epoch_start
        ) / 60

        print()
        print(
            f"Train loss       : {train_loss:.5f}"
        )

        print(
            f"Validation loss  : {val_loss:.5f}"
        )

        print(
            f"Val Macro AUROC  : {macro_auroc:.5f}"
        )

        print(
            f"Epoch time       : {epoch_minutes:.1f} min"
        )

        print()
        print("Validation AUROC by disease")
        print("-" * 78)

        for disease, auc in zip(
            CLASSES,
            class_aurocs,
        ):
            print(
                f"{disease:20}: {auc:.4f}"
            )

        # -------------------------------------------------
        # Save history
        # -------------------------------------------------

        append_history(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            macro_auroc=macro_auroc,
            learning_rate=current_lr,
            class_aurocs=class_aurocs,
        )

        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

        improved = macro_auroc > best_macro_auroc

        if improved:

            best_macro_auroc = macro_auroc
            epochs_without_improvement = 0

            save_checkpoint(
                path=BEST_MODEL_PATH,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_macro_auroc=best_macro_auroc,
                epochs_without_improvement=epochs_without_improvement,
            )

            print()
            print(
                "★ New best model saved"
                f" | Macro AUROC = {best_macro_auroc:.5f}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                "No validation AUROC improvement"
                f" ({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

        # Scheduler uses validation AUROC.
        scheduler.step(macro_auroc)

        # -------------------------------------------------
        # Always save last checkpoint
        # -------------------------------------------------

        save_checkpoint(
            path=LAST_CHECKPOINT_PATH,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            best_macro_auroc=best_macro_auroc,
            epochs_without_improvement=epochs_without_improvement,
        )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print("=" * 78)
            print(
                "EARLY STOPPING TRIGGERED"
            )
            print("=" * 78)

            break

    print()
    print("=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)

    print(
        "Best validation Macro AUROC:",
        f"{best_macro_auroc:.5f}"
    )

    print(
        "Best checkpoint:",
        BEST_MODEL_PATH
    )

    print(
        "Last checkpoint:",
        LAST_CHECKPOINT_PATH
    )

    print(
        "Training history:",
        HISTORY_PATH
    )

    print("=" * 78)


if __name__ == "__main__":
    main()