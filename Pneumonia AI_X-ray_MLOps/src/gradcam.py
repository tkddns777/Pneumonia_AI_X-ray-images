import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms, datasets
from pathlib import Path

# =====================================================
# Config
# =====================================================
MODEL_PATH = "models/resnet18_best.pth"
TEST_DIR   = "data/test"
OUT_DIR    = "results/gradcam"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# =====================================================
# Unicode-safe image save
# =====================================================
def imwrite_unicode(path, img_bgr):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()
    if ext == "":
        ext = ".png"
        path = path.with_suffix(ext)

    img_bgr = np.ascontiguousarray(img_bgr)

    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr,0,255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, img_bgr)

    if not ok:
        raise RuntimeError("cv2.imencode failed")

    buf.tofile(str(path))

    return str(path.resolve())


# =====================================================
# GradCAM
# =====================================================
class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fh = target_layer.register_forward_hook(self.forward_hook)
        self.bh = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):

        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):

        self.gradients = grad_out[0]

    def remove(self):

        self.fh.remove()
        self.bh.remove()

    def __call__(self, x):

        self.model.zero_grad()

        logits = self.model(x)
        probs = torch.softmax(logits,dim=1)

        pred = torch.argmax(probs,dim=1).item()
        prob = probs[0,pred].item()

        score = logits[0,pred]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads,dim=(2,3),keepdim=True)

        cam = torch.sum(weights*acts,dim=1)

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()[0]

        cam -= cam.min()
        cam /= cam.max()+1e-8

        return cam,pred,prob


# =====================================================
# Overlay
# =====================================================
def overlay(orig_rgb,cam,save_path,text):

    h,w = orig_rgb.shape[:2]

    cam = cv2.resize(cam,(w,h))

    heatmap = np.uint8(cam*255)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(orig_rgb,cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(orig_bgr,0.6,heatmap,0.4,0)

    cv2.putText(
        overlay,
        text,
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    imwrite_unicode(save_path,overlay)


# =====================================================
# Main
# =====================================================
def main():

    os.makedirs(OUT_DIR,exist_ok=True)

    # ---------------------------------
    # Load model
    # ---------------------------------

    checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)

    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features,num_classes)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Model loaded")

    # ---------------------------------
    # Dataset
    # ---------------------------------

    dataset = datasets.ImageFolder(TEST_DIR)

    print("Classes:",dataset.classes)
    print("Total images:",len(dataset))

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # ---------------------------------
    # GradCAM
    # ---------------------------------

    target_layer = model.layer4[-1]

    gradcam = GradCAM(model,target_layer)

    # ---------------------------------
    # Loop entire dataset
    # ---------------------------------

    for img_path,label in dataset.samples:

        try:

            img = Image.open(img_path).convert("RGB")
            orig_np = np.array(img)

            x = preprocess(img).unsqueeze(0).to(DEVICE)

            cam,pred,prob = gradcam(x)

            gt_name = dataset.classes[label]
            pred_name = dataset.classes[pred]

            filename = os.path.basename(img_path)

            text = f"GT:{gt_name} Pred:{pred_name} ({prob:.3f})"

            # ---------------------------------
            # Correct / Wrong
            # ---------------------------------

            if pred == label:

                save_dir = os.path.join(OUT_DIR,"correct")

            else:

                save_dir = os.path.join(OUT_DIR,"wrong")

            os.makedirs(save_dir,exist_ok=True)

            # ---------------------------------
            # False Positive / False Negative
            # ---------------------------------

            if pred != label:

                if gt_name == "NORMAL" and pred_name == "PNEUMONIA":

                    save_dir = os.path.join(OUT_DIR,"false_positive")

                elif gt_name == "PNEUMONIA" and pred_name == "NORMAL":

                    save_dir = os.path.join(OUT_DIR,"false_negative")

                os.makedirs(save_dir,exist_ok=True)

            save_path = os.path.join(
                save_dir,
                f"GT_{gt_name}_PRED_{pred_name}_{filename}"
            )

            overlay(orig_np,cam,save_path,text)

            print("Saved:",save_path)

        except Exception as e:

            print("Error:",img_path)
            print(e)

    gradcam.remove()

    print("\nGradCAM generation finished")


# =====================================================
if __name__ == "__main__":
    main()