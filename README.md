# Explainable AI for Diabetic Retinopathy Screening

An AI-powered diabetic retinopathy screening application that analyzes retinal fundus images and predicts the severity of diabetic retinopathy using a fine-tuned EfficientNet-B0 deep learning model.

The system combines **deep learning, explainable AI, FastAPI, Streamlit, SQLite, Grad-CAM visualization, image-quality validation, and PDF report generation** into a complete end-to-end screening application.

> **Disclaimer:** This project is an educational and research prototype. It is not a medical diagnostic device and should not be used as a substitute for examination by a qualified ophthalmologist.

---

## Project Overview

Diabetic retinopathy (DR) is a diabetes-related eye disease that can lead to vision loss if it is not detected and managed appropriately.

This project demonstrates how deep learning can be used to classify retinal fundus images into five diabetic retinopathy severity levels.

The system accepts a retinal image, performs an image-quality check, sends the image to an AI model through a FastAPI backend, generates a prediction and confidence score, and provides a Grad-CAM visualization showing the regions that influenced the model's prediction.

The application also provides patient information management, screening history, SQLite database storage, and downloadable PDF screening reports.

---

## Key Features

- Retinal fundus image upload
- Image-quality validation before AI prediction
- Five-class diabetic retinopathy classification
- EfficientNet-B0 deep learning model
- ImageNet-pretrained backbone with fine-tuning
- Confidence score and probability distribution
- Clinical-style severity interpretation
- Grad-CAM explainability
- Attention overlay visualization
- Patient information form
- Screening history
- SQLite database storage
- PDF screening report generation
- Streamlit interactive frontend
- FastAPI backend
- REST API for model prediction
- Session-state persistence
- Professional dark medical/AI interface
- Low-confidence warning
- Input validation and robustness checks

---

## DR Severity Classes

The model predicts five severity categories:

| Class | Severity |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Model

The project uses **EfficientNet-B0** as the backbone.

The model is initialized with ImageNet-pretrained weights and fine-tuned for five-class diabetic retinopathy classification.

### Why EfficientNet-B0?

EfficientNet-B0 provides a good balance between:

- Classification performance
- Computational efficiency
- Model size
- Inference speed

This makes it suitable for an educational screening application running on a normal computer.

### Model Architecture

```text
Input Retinal Image
        |
        v
Image Preprocessing
        |
        v
EfficientNet-B0
(ImageNet Pretrained)
        |
        v
Feature Extraction
        |
        v
Classification Layer
        |
        v
5-Class Prediction
        |
        +------------------+
        |                  |
        v                  v
 Prediction           Grad-CAM
        |                  |
        v                  v
 Severity             Attention
 Confidence           Visualization