"""
Evaluate the trained model on the held-out TEST split.

The split is exactly the same 70% / 15% / 15%
split used during training.

Usage:
    python model/evaluate.py
"""

import os

import torch
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

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
# Main
# ==================================================

def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("==============================")
    print("DR MODEL TEST EVALUATION")
    print("==============================")
    print(f"Device: {device}")
    print()


    # ==================================================
    # Load dataset
    # ==================================================

    print("Loading dataset...")

    full_dataset = APTOSDataset(
        image_size=224,
        train=False
    )

    total_size = len(full_dataset)

    print(
        f"Total images: {total_size}"
    )


    # ==================================================
    # Recreate EXACT 70 / 15 / 15 split
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


    generator = torch.Generator().manual_seed(42)


    train_subset, val_subset, test_subset = random_split(
        full_dataset,
        [
            train_size,
            val_size,
            test_size
        ],
        generator=generator
    )


    print()
    print("Dataset split:")
    print(
        f"Training   : {len(train_subset)}"
    )
    print(
        f"Validation : {len(val_subset)}"
    )
    print(
        f"Test       : {len(test_subset)}"
    )

    print()


    # ==================================================
    # Test DataLoader
    # ==================================================

    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )


    # ==================================================
    # Load trained model
    # ==================================================

    checkpoint_path = os.path.join(
        "model",
        "best_model.pth"
    )


    print("Loading model...")

    if not os.path.exists(
        checkpoint_path
    ):

        print(
            f"ERROR: Model checkpoint not found:"
        )

        print(
            checkpoint_path
        )

        return


    model = DRModel(
        num_classes=5
    ).to(device)


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint
    )

    model.eval()


    print(
        "Model loaded successfully."
    )

    print()


    # ==================================================
    # Run test predictions
    # ==================================================

    all_preds = []
    all_labels = []


    print(
        "Running TEST evaluation..."
    )


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device
            )

            outputs = model(
                images
            )

            preds = outputs.argmax(
                dim=1
            ).cpu().numpy()


            all_preds.extend(
                preds
            )

            all_labels.extend(
                labels.numpy()
            )


    # ==================================================
    # Calculate metrics
    # ==================================================

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )


    kappa = cohen_kappa_score(
        all_labels,
        all_preds,
        weights="quadratic"
    )


    # ==================================================
    # Print results
    # ==================================================

    print()
    print("==============================")
    print("TEST RESULTS")
    print("==============================")


    print(
        f"Test Accuracy: {accuracy:.4f}"
    )


    print(
        f"Quadratic Weighted Kappa: {kappa:.4f}"
    )


    print()


    # ==================================================
    # Classification report
    # ==================================================

    print(
        "Classification Report"
    )

    print(
        "=============================="
    )


    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )


    print(
        report
    )


    # ==================================================
    # Confusion matrix
    # ==================================================

    cm = confusion_matrix(
        all_labels,
        all_preds
    )


    print(
        "Confusion Matrix"
    )

    print(
        "=============================="
    )

    print(
        cm
    )

    print()


    # ==================================================
    # Save confusion matrix
    # ==================================================

    os.makedirs(
        "report",
        exist_ok=True
    )


    fig, ax = plt.subplots(
        figsize=(8, 7)
    )


    im = ax.imshow(
        cm,
        cmap="Blues"
    )


    ax.set_xticks(
        range(
            len(CLASS_NAMES)
        )
    )


    ax.set_yticks(
        range(
            len(CLASS_NAMES)
        )
    )


    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right"
    )


    ax.set_yticklabels(
        CLASS_NAMES
    )


    ax.set_xlabel(
        "Predicted"
    )


    ax.set_ylabel(
        "True"
    )


    ax.set_title(
        "Test Confusion Matrix"
    )


    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                )
            )


    fig.colorbar(
        im
    )


    fig.tight_layout()


    output_path = os.path.join(
        "report",
        "test_confusion_matrix.png"
    )


    fig.savefig(
        output_path,
        dpi=150
    )


    plt.close(
        fig
    )


    print(
        f"Saved test confusion matrix to:"
    )

    print(
        output_path
    )


    # ==================================================
    # Complete
    # ==================================================

    print()

    print("==============================")

    print(
        "TEST EVALUATION COMPLETE"
    )

    print(
        "=============================="
    )

    print()


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":
    main()