"""
Grad-CAM heatmap generation for the DR model.

Uses the pytorch-grad-cam library so we don't hand-roll gradient hooks:
    pip install pytorch-grad-cam

Produces a heatmap showing which region of the retina image most
influenced the model's predicted severity grade.
"""

import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from model import DRModel
from dataset import preprocess_image, get_eval_transforms


def load_model_for_cam(checkpoint_path, device):
    model = DRModel(num_classes=5).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def generate_gradcam(model, image_path, device, image_size=224, target_class=None):
    """
    Returns:
        overlay: uint8 RGB image with the heatmap overlaid (for display)
        pred_class: predicted severity grade (int)
        probs: softmax probabilities per class
    """
    raw_image = preprocess_image(image_path, image_size)  # RGB, uint8, HxWx3
    rgb_float = raw_image.astype(np.float32) / 255.0

    transform = get_eval_transforms()
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # forward pass for prediction (needs grad enabled for Grad-CAM, so no no_grad here)
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]
    pred_class = int(probs.argmax())

    cam_target_class = target_class if target_class is not None else pred_class

    cam = GradCAM(model=model, target_layers=[model.target_layer])
    grayscale_cam = cam(input_tensor=input_tensor,
                         targets=None if target_class is None else None)[0]
    # pytorch-grad-cam targets the highest-scoring class by default when targets=None

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return overlay, pred_class, probs


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_cam(args.checkpoint, device)
    overlay, pred_class, probs = generate_gradcam(model, args.image, device)

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    print(f"Predicted: {class_names[pred_class]} (confidence {probs[pred_class]:.2%})")

    plt.imshow(overlay)
    plt.title(f"Grad-CAM: {class_names[pred_class]}")
    plt.axis("off")
    plt.savefig("gradcam_output.png", dpi=150, bbox_inches="tight")
    print("Saved gradcam_output.png")
