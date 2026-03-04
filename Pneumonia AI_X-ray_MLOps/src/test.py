import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)


# =====================================================
# Device
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Load trained model
# =====================================================
MODEL_PATH = r"C:\Users\user\OneDrive\바탕 화면\코딩\Pneumonia_AI_X-ray\Pneumonia AI_X-ray_MLOps\models\resnet18_best.pth"

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names = checkpoint["class_names"]

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, len(class_names))
)

model.to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(
    f"✅ Model loaded "
    f"(epoch={checkpoint['epoch']}, "
    f"test_acc={checkpoint['test_accuracy']:.3f})"
)

# =====================================================
# Test dataset
# =====================================================
TEST_DIR = r"C:\Users\user\OneDrive\바탕 화면\코딩 데이터\Pneumonia CT images\test"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print("Test classes:", test_dataset.classes)

# =====================================================
# Prediction
# =====================================================
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)[:,1]  # pneumonia probability
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# =====================================================
# Metrics
# =====================================================
acc = accuracy_score(all_labels, all_preds)

cm = confusion_matrix(all_labels, all_preds)

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

auroc = roc_auc_score(all_labels, all_probs)
auprc = average_precision_score(all_labels, all_probs)

print("\n===== Evaluation Results =====")

print(f"Accuracy     : {acc:.3f}")
print(f"AUROC        : {auroc:.3f}")
print(f"AUPRC        : {auprc:.3f}")
print(f"Sensitivity  : {sensitivity:.3f}")
print(f"Specificity  : {specificity:.3f}")

# =====================================================
# Confusion Matrix
# =====================================================
plt.figure(figsize=(4,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# =====================================================
# ROC Curve
# =====================================================
fpr, tpr, _ = roc_curve(all_labels, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("results/roc_curve.png")
plt.close()

# =====================================================
# Precision-Recall Curve
# =====================================================
precision, recall, _ = precision_recall_curve(all_labels, all_probs)

plt.figure()
plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("results/precision_recall_curve.png")
plt.close()