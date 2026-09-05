"""
Dataset and preprocessing for the APTOS 2019 diabetic retinopathy dataset.

This version loads the smaller 224x224 APTOS dataset directly
from Hugging Face instead of requiring the original 8+ GB image files.
"""

import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_black_border(image, tol=7):
    """Crop the black border around a circular fundus image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    mask = gray > tol

    if mask.sum() == 0:
        return image

    coords = np.argwhere(mask)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return image[y0:y1, x0:x1]


def apply_clahe(image):
    """Apply CLAHE to improve retinal image contrast."""

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(image, image_size=224):
    """
    Preprocess an image received from Hugging Face.

    Hugging Face provides the image as a PIL image.
    """

    image = np.array(image.convert("RGB"))

    image = crop_black_border(image)

    image = apply_clahe(image)

    image = cv2.resize(
        image,
        (image_size, image_size)
    )

    return image


def get_train_transforms():

    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.RandomHorizontalFlip(),

        transforms.RandomVerticalFlip(),

        transforms.RandomRotation(20),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        ),
    ])


def get_eval_transforms():

    return transforms.Compose([
        transforms.ToPILImage(),

        transforms.ToTensor(),

        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD
        ),
    ])


class APTOSDataset(Dataset):

    def __init__(
        self,
        csv_path=None,
        images_dir=None,
        image_size=224,
        train=True
    ):

        print("Loading APTOS dataset from Hugging Face...")

        self.dataset = load_dataset(
            "bumbledeep/aptos",
            split="train"
        )

        self.image_size = image_size

        self.transform = (
            get_train_transforms()
            if train
            else get_eval_transforms()
        )

        print(
            f"Loaded {len(self.dataset)} images."
        )

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, idx):

        row = self.dataset[idx]

        image = row["image"]

        label = int(row["label_code"])

        image = preprocess_image(
            image,
            self.image_size
        )

        image = self.transform(image)

        return image, label

    def class_counts(self):

        labels = [
            int(x)
            for x in self.dataset["label_code"]
        ]

        counts = np.bincount(
            labels,
            minlength=5
        )

        return counts


def compute_class_weights(
    dataset,
    num_classes=5
):

    counts = dataset.class_counts()

    total = counts.sum()

    weights = total / (
        num_classes *
        np.maximum(counts, 1)
    )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )