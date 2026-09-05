"""
Train Version 2 of the Diabetic Retinopathy severity classifier.

Dataset:
    APTOS 2019

Split:
    70% Training
    15% Validation
    15% Test

Important:
    The test set is NOT used during training.

Version 2 improvements:
    - Exact reproducible 70/15/15 split
    - Class weights calculated from TRAINING data only
    - Saves the model separately as best_model_v2.pth
    - Saves split indices for reproducibility
    - Tracks validation accuracy and Quadratic Weighted Kappa
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score
)

from tqdm import tqdm

from dataset import APTOSDataset
from model import DRModel


# ==================================================
# Class names
# ==================================================

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]


# ==================================================
# Arguments
# ==================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224
    )

    parser.add_argument(
        "--output",
        type=str,
        default="model/best_model_v2.pth"
    )

    return parser.parse_args()


# ==================================================
# Train one epoch
# ==================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    for images, labels in tqdm(
        loader,
        desc="train",
        leave=False
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    return avg_loss, accuracy


# ==================================================
# Validation
# ==================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []

    for images, labels in tqdm(
        loader,
        desc="val",
        leave=False
    ):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item()
            * images.size(0)
        )

        preds = outputs.argmax(
            dim=1
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    kappa = cohen_kappa_score(
        all_labels,
        all_preds,
        weights="quadratic"
    )

    return (
        avg_loss,
        accuracy,
        kappa
    )


# ==================================================
# Calculate training class weights
# ==================================================

def compute_train_class_weights(
    dataset,
    indices,
    num_classes=5
):

    labels = []

    print()
    print("Calculating class distribution from training data...")

    for idx in indices:

        row = dataset.dataset[idx]

        label = int(
            row["label_code"]
        )

        labels.append(label)

    labels = np.array(
        labels,
        dtype=np.int64
    )

    counts = np.bincount(
        labels,
        minlength=num_classes
    )

    total = counts.sum()

    weights = total / (
        num_classes
        * np.maximum(counts, 1)
    )

    return (
        counts,
        torch.tensor(
            weights,
            dtype=torch.float32
        )
    )


# ==================================================
# Main
# ==================================================

def main():

    args = parse_args()


    # ==================================================
    # Device
    # ==================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()

    print("==============================")
    print("DR MODEL TRAINING - VERSION 2")
    print("==============================")

    print(
        f"Device: {device}"
    )

    print()


    # ==================================================
    # Load dataset
    # ==================================================

    print("Loading dataset...")

    full_dataset = APTOSDataset(
        image_size=args.image_size,
        train=True
    )

    total_size = len(
        full_dataset
    )

    print(
        f"Total images: {total_size}"
    )


    # ==================================================
    # 70 / 15 / 15 split
    # ==================================================

    train_size = int(
        total_size * 0.70
    )

    val_size = int(
        total_size * 0.15
    )

    test_size = (
        total_size
        - train_size
        - val_size
    )


    generator = torch.Generator().manual_seed(
        42
    )


    train_indices, val_indices, test_indices = random_split(
        range(total_size),
        [
            train_size,
            val_size,
            test_size
        ],
        generator=generator
    )


    print()

    print("==============================")
    print("DATASET SPLIT")
    print("==============================")

    print(
        f"Training   : {len(train_indices)}"
    )

    print(
        f"Validation : {len(val_indices)}"
    )

    print(
        f"Test       : {len(test_indices)}"
    )


    # ==================================================
    # Save split indices
    # ==================================================

    os.makedirs(
        "model",
        exist_ok=True
    )


    split_path = os.path.join(
        "model",
        "dataset_split_v2.pth"
    )


    torch.save(
        {
            "train_indices": train_indices.indices,
            "val_indices": val_indices.indices,
            "test_indices": test_indices.indices
        },
        split_path
    )


    print()

    print(
        f"Saved split information to:"
    )

    print(
        split_path
    )


    # ==================================================
    # Create train/evaluation datasets
    # ==================================================

    print()

    print("Creating datasets...")


    train_dataset = APTOSDataset(
        image_size=args.image_size,
        train=True
    )


    eval_dataset = APTOSDataset(
        image_size=args.image_size,
        train=False
    )


    train_ds = torch.utils.data.Subset(
        train_dataset,
        train_indices.indices
    )


    val_ds = torch.utils.data.Subset(
        eval_dataset,
        val_indices.indices
    )


    test_ds = torch.utils.data.Subset(
        eval_dataset,
        test_indices.indices
    )


    # ==================================================
    # DataLoaders
    # ==================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )


    # ==================================================
    # Class weights
    # ==================================================

    train_counts, class_weights = compute_train_class_weights(
        train_dataset,
        train_indices.indices
    )


    class_weights = class_weights.to(
        device
    )


    print()

    print("==============================")
    print("TRAINING CLASS DISTRIBUTION")
    print("==============================")


    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{name:<20}: {train_counts[i]}"
        )


    print()

    print(
        "Class weights:"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        print(
            f"{name:<20}: {class_weights[i].item():.4f}"
        )


    # ==================================================
    # Model
    # ==================================================

    print()

    print(
        "Creating EfficientNet-B0..."
    )


    model = DRModel(
        num_classes=5
    ).to(device)


    # ==================================================
    # Loss
    # ==================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # ==================================================
    # Optimizer
    # ==================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )


    # ==================================================
    # Learning-rate scheduler
    # ==================================================

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=2,
            factor=0.5
        )
    )


    # ==================================================
    # Training
    # ==================================================

    best_kappa = -1.0

    best_state = None


    print()

    print("==============================")
    print("STARTING TRAINING")
    print("==============================")


    for epoch in range(
        1,
        args.epochs + 1
    ):

        print()

        print(
            f"Epoch {epoch}/{args.epochs}"
        )


        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )


        # --------------------------------------------------
        # Validate
        # --------------------------------------------------

        val_loss, val_acc, val_kappa = validate(
            model,
            val_loader,
            criterion,
            device
        )


        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        scheduler.step(
            val_kappa
        )


        current_lr = optimizer.param_groups[0]["lr"]


        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print()

        print(
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f}"
        )

        print(
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_kappa={val_kappa:.4f}"
        )

        print(
            f"learning_rate={current_lr:.6f}"
        )


        # --------------------------------------------------
        # Save best model
        # --------------------------------------------------

        if val_kappa > best_kappa:

            best_kappa = val_kappa

            best_state = copy.deepcopy(
                model.state_dict()
            )


            torch.save(
                best_state,
                args.output
            )


            print()

            print(
                "--> Saved new best V2 model"
            )

            print(
                f"    Validation QWK = {best_kappa:.4f}"
            )


    # ==================================================
    # Training complete
    # ==================================================

    print()

    print("==============================")
    print("VERSION 2 TRAINING COMPLETE")
    print("==============================")


    print(
        f"Best validation QWK: {best_kappa:.4f}"
    )


    print()

    print(
        f"Model saved to:"
    )

    print(
        args.output
    )


    print()

    print(
        "Test set was NOT used during training."
    )

    print(
        f"Test images reserved: {len(test_ds)}"
    )


    print()

    print(
        "Next step:"
    )

    print(
        "Evaluate best_model_v2.pth on the test set."
    )


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":
    main()