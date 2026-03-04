# Pneumonia AI Diagnosis System

Deep Learning 기반 Chest X-ray Pneumonia Detection 시스템입니다.  
ResNet18 모델을 이용하여 폐렴을 분류하고, Grad-CAM을 통해 모델의 판단 근거를 시각화하며, FastAPI와 Streamlit을 이용한 웹 기반 AI 진단 인터페이스를 제공합니다.

---

# Project Overview

본 프로젝트는 의료 영상 AI 모델을 학습하고 평가하는 것뿐만 아니라 실제 서비스 형태로 활용할 수 있도록 **AI inference API와 Web UI까지 포함한 의료 AI 시스템 프로토타입**을 구현하는 것을 목표로 합니다.

주요 기능:

- Chest X-ray Pneumonia Classification
- Explainable AI (Grad-CAM)
- Model evaluation (AUROC, AUPRC, Sensitivity, Specificity)
- FastAPI 기반 inference API
- Streamlit 기반 Web UI
- Grad-CAM heatmap 시각화

---

# System Architecture

---

# Features

## 1. Pneumonia Classification

ResNet18 기반 CNN 모델을 사용하여 Chest X-ray 이미지에서 폐렴 여부를 분류합니다.

Output:

- NORMAL
- PNEUMONIA

---

## 2. Explainable AI (Grad-CAM)

모델이 폐렴으로 판단한 **영상 영역을 heatmap 형태로 시각화**합니다.

Grad-CAM은 모델의 마지막 convolution layer에서 activation과 gradient를 이용하여 생성됩니다.

Example:

Original X-ray → Grad-CAM heatmap overlay

---

## 3. Model Evaluation

모델 성능은 다음 지표를 이용하여 평가합니다.

- Accuracy
- AUROC
- AUPRC
- Sensitivity
- Specificity
- Confusion Matrix

---

## 4. FastAPI Inference Server

FastAPI를 이용하여 AI 모델을 API 형태로 제공합니다.

Endpoint:
