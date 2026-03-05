import os
import glob
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json

# =====================================================
# 설정
# =====================================================

MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia models\resnet18_seed0_epoch002_acc0.960.pth"
TEST_DIR   = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia X-ray images\test"
OUT_DIR    = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia X-ray images\Grad-CAM"
JSON_DIR   = r"C:\Users\user\OneDrive\바탕 화면\코딩\Pneumonia_AI_X-ray\Pneumonia AI_X-ray\json_analysis"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Image_SIZE = 224

os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================
# Unicode-safe 저장
def save_json_safe(path, data):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(data, indent=4, ensure_ascii=False)

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)

    print("[JSON SAVED]", path)

# =====================================================
# Unicode-safe 저장
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
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(ext, img_bgr)

    if not ok:
        raise RuntimeError("cv2.imencode failed")

    buf.tofile(str(path))

    return str(path.resolve())


# =====================================================
# Grad-CAM
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

    def __call__(self, input_tensor):

        self.model.zero_grad()

        logits = self.model(input_tensor)

        probs = torch.softmax(logits, dim=1)

        pred_idx = int(torch.argmax(probs))

        pred_prob = float(probs[0, pred_idx])

        score = logits[0, pred_idx]

        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=(2,3), keepdim=True)

        cam = torch.sum(weights * acts, dim=1)

        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()[0]

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam, pred_idx, pred_prob


# =====================================================
# GradCAM hotspot 분석
# =====================================================

def analyze_heatmap(cam):

    threshold = 0.6

    mask = cam > threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:

        return {
            "center_x":0,
            "center_y":0,
            "area_ratio":0
        }

    center_x = float(xs.mean() / cam.shape[1])
    center_y = float(ys.mean() / cam.shape[0])

    area_ratio = float(mask.sum() / mask.size)

    return {
        "center_x":center_x,
        "center_y":center_y,
        "area_ratio":area_ratio
    }

# =====================================================
# 폐 영역 분석
def get_lung_region(center_x, center_y):

    # 좌우 폐
    if center_x < 0.5:
        side = "Left Lung"
    else:
        side = "Right Lung"

    # 상중하
    if center_y < 0.33:
        level = "Upper"
    elif center_y < 0.66:
        level = "Middle"
    else:
        level = "Lower"

    return f"{side} {level}"

# =====================================================
# Grad-CAM 의심구역 박스화
def get_bbox_from_heatmap(cam, threshold=0.6):
    """
    Grad-CAM heatmap에서 activation 영역 bounding box 계산
    cam: (H,W) 0~1 heatmap
    return: (x_min, y_min, x_max, y_max)
    """

    mask = cam > threshold

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    return (x_min, y_min, x_max, y_max)

# =====================================================
# overlay 생성
# =====================================================

def overlay_and_save(orig_rgb, cam, save_path, text, bbox=None):

    h, w = orig_rgb.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))

    heatmap = np.uint8(cam_resized * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap, 0.4, 0)

    # bounding box
    if bbox is not None:

        x_min, y_min, x_max, y_max = bbox

        # heatmap 크기
        cam_h, cam_w = cam.shape

        # 좌표 변환
        x_min = int(x_min * w / cam_w)
        x_max = int(x_max * w / cam_w)
        y_min = int(y_min * h / cam_h)
        y_max = int(y_max * h / cam_h)

        cv2.rectangle(
            overlay,
            (x_min, y_min),
            (x_max, y_max),
            (0,255,0),
            3
        )

    cv2.putText(
        overlay,
        text,
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255,255,255),
        2
    )

    imwrite_unicode(save_path, overlay)

    return overlay


# =====================================================
# 메인 실행
# =====================================================

def main():

    checkpoint = torch.load(MODEL_PATH,map_location=DEVICE)

    state_dict = checkpoint["model_state_dict"]

    class_names = checkpoint.get("class_names",["NORMAL","PNEUMONIA"])

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # 필요시 여기에 활성화
        nn.Linear(model.fc.in_features, len(class_names))
    )

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([

        transforms.Resize((Image_SIZE,Image_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    target_layer = model.layer4[-1]

    gradcam = GradCAM(model,target_layer)

    for cname in class_names:

        class_folder = os.path.join(TEST_DIR,cname)

        if not os.path.isdir(class_folder):
            continue

        images = []

        for ext in IMG_EXTS:
            images.extend(glob.glob(os.path.join(class_folder,f"*{ext}")))

        if len(images)==0:
            continue

        img_path = random.choice(images)

        img = Image.open(img_path).convert("RGB")

        orig_np = np.array(img)

        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        cam,pred_idx,pred_prob = gradcam(input_tensor)

        bbox = get_bbox_from_heatmap(cam)

        pred_name = class_names[pred_idx]

        # ===============================
        # hotspot 분석
        # ===============================

        hotspot = analyze_heatmap(cam)

        center_x = hotspot["center_x"]
        center_y = hotspot["center_y"]

        region = get_lung_region(center_x, center_y)

        # ===============================
        # LLM용 structured data
        # ===============================

        result = {

            "image":os.path.basename(img_path),

            "ground_truth":cname,

            "prediction":pred_name,

            "confidence":float(pred_prob),

            "activation_center_x":hotspot["center_x"],

            "activation_center_y":hotspot["center_y"],

            "activation_area_ratio":hotspot["area_ratio"],

            "activation_region": region,

            "bbox": bbox
        }

        # ===============================
        # JSON 저장
        # ===============================

        json_path = Path(JSON_DIR) / f"{Path(img_path).stem}_analysis.json"

        print("Saving JSON to:", json_path)

        save_json_safe(json_path, result)


        # overlay 저장

        text = f"GT:{cname} Pred:{pred_name} ({pred_prob:.3f})"

        overlay_path = os.path.join(
            OUT_DIR,
            f"{Path(img_path).stem}_gradcam.png"
        )

        overlay_and_save(orig_np,cam,overlay_path,text,bbox)

        print("\n===== GradCAM Analysis =====")
        print(json.dumps(result,indent=4))

    gradcam.remove()

    print("\nDONE")


if __name__ == "__main__":

    main()