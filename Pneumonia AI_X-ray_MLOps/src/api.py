from fastapi import FastAPI, UploadFile, File
import torch
import numpy as np
import cv2
import base64
from PIL import Image
from torchvision import transforms

from src.model_loader import load_model
from src.gradcam_utils import GradCAM

app = FastAPI(
    title="Pneumonia AI Diagnosis API",
    description="Deep Learning X-ray Pneumonia Detection with Grad-CAM Explainability",
    version="1.0.0",
    contact={
        "name": "SangUn Kim",
        "email": "example@email.com"
    }
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, class_names = load_model()

gradcam = GradCAM(model, model.layer4[-1])

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def create_gradcam_overlay(orig_rgb, cam):

    h,w = orig_rgb.shape[:2]

    cam = cv2.resize(cam,(w,h))

    heatmap = np.uint8(cam*255)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(orig_rgb,cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(orig_bgr,0.6,heatmap,0.4,0)

    return overlay


def image_to_base64(img):

    _,buffer = cv2.imencode(".png",img)

    return base64.b64encode(buffer).decode()


@app.get("/")
def root():

    return {"message":"Pneumonia AI API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    orig_np = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    cam, pred, prob = gradcam(input_tensor)

    overlay = create_gradcam_overlay(orig_np, cam)

    gradcam_base64 = image_to_base64(overlay)

    return {
        "prediction": class_names[pred],
        "probability": float(prob),
        "gradcam_image": gradcam_base64
    }