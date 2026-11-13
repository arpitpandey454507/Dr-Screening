import os
import sys
import base64
import hashlib
import io
import re
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape

import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
)

# ============================================================
# PATHS & DIRECT MODEL / DATABASE IMPORTS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
BACKEND_DIR = ROOT_DIR / "backend"

for _path in (str(MODEL_DIR), str(BACKEND_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from model import DRModel
from dataset import get_eval_transforms, crop_black_border, apply_clahe
from database import (
    initialize_database,
    save_screening,
    get_all_screenings,
    get_screening,
    delete_screening,
)

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]
IMAGE_SIZE = 224

# Checks model/best_model_v3.pth first, falls back to root best_model.pth
CHECKPOINT_PATH = (
    MODEL_DIR / "best_model_v3.pth"
    if (MODEL_DIR / "best_model_v3.pth").exists()
    else ROOT_DIR / "best_model.pth"
)

DEVELOPER_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "arpit_pandey.png"


# ============================================================
# CACHED MODEL & DATABASE INITIALIZATION
# ============================================================

@st.cache_resource
def init_db():
    initialize_database()
    return True


@st.cache_resource
def get_dr_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT_PATH.exists():
        return None, device, f"Checkpoint not found at {CHECKPOINT_PATH}"

    try:
        loaded_model = DRModel(num_classes=5).to(device)
        checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        loaded_model.load_state_dict(state_dict)
        loaded_model.eval()
        return loaded_model, device, None
    except Exception as e:
        return None, device, str(e)


def preprocess_uploaded_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = crop_black_border(image)
    image = apply_clahe(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image


def encode_overlay_to_base64(overlay_rgb_uint8: np.ndarray) -> str:
    image = Image.fromarray(overlay_rgb_uint8)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_local_prediction(image_bytes: bytes) -> dict:
    model, device, load_error = get_dr_model()
    if model is None:
        raise RuntimeError(f"Model could not be loaded: {load_error}")

    processed_image = preprocess_uploaded_image(image_bytes)
    transform = get_eval_transforms()
    input_tensor = transform(processed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities_tensor = torch.softmax(outputs, dim=1)
        probabilities = probabilities_tensor.cpu().numpy()[0]

    predicted_class = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = float(probabilities[predicted_class])
    probability_dict = {
        CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))
    }

    heatmap_base64 = None
    try:
        cam = GradCAM(model=model, target_layers=[model.target_layer])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        rgb_float = processed_image.astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
        heatmap_base64 = encode_overlay_to_base64(overlay)
    except Exception:
        heatmap_base64 = None

    return {
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probability_dict,
        "heatmap_base64": heatmap_base64,
    }


def pil_to_data_uri(image, image_format="PNG"):
    """Convert a PIL image to a browser-safe data URI for styled HTML display."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=image_format, optimize=True)
    encoded = base64.b
