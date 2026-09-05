import base64
import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from database import (
    initialize_database,
    save_screening,
    get_all_screenings,
    get_screening,
    delete_screening,
)

# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "model"
    )
)

sys.path.insert(0, MODEL_DIR)

from model import DRModel
from dataset import (
    get_eval_transforms,
    crop_black_border,
    apply_clahe
)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

IMAGE_SIZE = 224

CHECKPOINT_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "model",
        "best_model_v3.pth"
    )
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = None


# ============================================================
# SCREENING REQUEST MODEL
# ============================================================

class ScreeningCreate(BaseModel):
    patient_name: str
    patient_id: str
    age: int
    gender: str
    screening_notes: str = ""
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    print("=" * 60)
    print("Loading Diabetic Retinopathy Model V3...")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Device:", device)
    print("=" * 60)

    if not os.path.exists(CHECKPOINT_PATH):

        print("ERROR: Model checkpoint not found.")
        print(CHECKPOINT_PATH)

        model = None

        return

    try:

        loaded_model = DRModel(
            num_classes=5
        ).to(device)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device
        )

        # ----------------------------------------------------
        # Handle different checkpoint formats
        # ----------------------------------------------------

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        # ----------------------------------------------------
        # Load weights
        # ----------------------------------------------------

        loaded_model.load_state_dict(
            state_dict
        )

        loaded_model.eval()

        model = loaded_model

        print("Model loaded successfully.")

    except Exception as e:

        model = None

        print(
            "ERROR while loading model:",
            e
        )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Initialize SQLite database
    initialize_database()

    # Load ML model
    load_model()

    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Diabetic Retinopathy Screening API",
    description=(
        "AI-assisted diabetic retinopathy screening "
        "API using DR Model V3."
    ),
    version="3.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_uploaded_image(
    image_bytes: bytes
) -> np.ndarray:

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = np.array(image)

    image = crop_black_border(
        image
    )

    image = apply_clahe(
        image
    )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    return image


# ============================================================
# ENCODE GRAD-CAM IMAGE
# ============================================================

def encode_overlay_to_base64(
    overlay_rgb_uint8: np.ndarray
) -> str:

    image = Image.fromarray(
        overlay_rgb_uint8
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "classes": CLASS_NAMES,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    try:

        image_bytes = await file.read()

        if not image_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read uploaded image: {e}"
            )
        )

    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    try:

        processed_image = (
            preprocess_uploaded_image(
                image_bytes
            )
        )

        transform = (
            get_eval_transforms()
        )

        input_tensor = (
            transform(
                processed_image
            )
            .unsqueeze(0)
            .to(device)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not process image: {e}"
            )
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities_tensor = (
                torch.softmax(
                    outputs,
                    dim=1
                )
            )

            probabilities = (
                probabilities_tensor
                .cpu()
                .numpy()[0]
            )

        predicted_class = int(
            np.argmax(
                probabilities
            )
        )

        predicted_label = (
            CLASS_NAMES[
                predicted_class
            ]
        )

        confidence = float(
            probabilities[
                predicted_class
            ]
        )

        probability_dict = {

            CLASS_NAMES[i]: float(
                probabilities[i]
            )

            for i in range(
                len(CLASS_NAMES)
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {e}"
            )
        )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    heatmap_base64: Optional[str] = None

    try:

        cam = GradCAM(
            model=model,
            target_layers=[
                model.target_layer
            ]
        )

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=None
        )[0]

        rgb_float = (
            processed_image.astype(
                np.float32
            ) / 255.0
        )

        overlay = show_cam_on_image(
            rgb_float,
            grayscale_cam,
            use_rgb=True
        )

        heatmap_base64 = (
            encode_overlay_to_base64(
                overlay
            )
        )

    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

    # --------------------------------------------------------
    # Return prediction
    # --------------------------------------------------------

    return {

        "predicted_class":
            predicted_class,

        "predicted_label":
            predicted_label,

        "confidence":
            confidence,

        "probabilities":
            probability_dict,

        "heatmap_base64":
            heatmap_base64,
    }


# ============================================================
# CREATE SCREENING
# ============================================================

@app.post("/screenings")
def create_screening(
    screening: ScreeningCreate
):

    try:

        screening_id = save_screening(

            patient_name=
                screening.patient_name,

            patient_id=
                screening.patient_id,

            age=
                screening.age,

            gender=
                screening.gender,

            screening_notes=
                screening.screening_notes,

            predicted_label=
                screening.predicted_label,

            confidence=
                screening.confidence,

            probabilities=
                screening.probabilities,
        )

        return {

            "success": True,

            "screening_id":
                screening_id,

            "message":
                "Screening saved successfully.",
        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Could not save screening: {e}"
            )
        )


# ============================================================
# GET ALL SCREENINGS
# ============================================================

@app.get("/screenings")
def screenings():

    try:

        return get_all_screenings()

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not load screening history: "
                f"{e}"
            )
        )


# ============================================================
# GET SINGLE SCREENING
# ============================================================

@app.get("/screenings/{screening_id}")
def screening_details(
    screening_id: int
):

    try:

        record = get_screening(
            screening_id
        )

        if record is None:

            raise HTTPException(

                status_code=404,

                detail=(
                    "Screening record not found."
                )
            )

        return record

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not load screening: "
                f"{e}"
            )
        )


# ============================================================
# DELETE SCREENING
# ============================================================

@app.delete("/screenings/{screening_id}")
def remove_screening(
    screening_id: int
):

    try:

        deleted = delete_screening(
            screening_id
        )

        if not deleted:

            raise HTTPException(

                status_code=404,

                detail=(
                    "Screening record not found."
                )
            )

        return {

            "success": True,

            "message":
                "Screening deleted successfully.",
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                "Could not delete screening: "
                f"{e}"
            )
        )