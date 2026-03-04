import os
import streamlit as st
import requests
import base64
from PIL import Image
import io
from streamlit_image_comparison import image_comparison

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Pneumonia AI", layout="wide")

st.title("🩺 Pneumonia AI Diagnosis")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original X-ray")
        st.image(image, use_column_width=True)

    if st.button("Run AI Diagnosis"):

        with st.spinner("Running AI inference..."):

            files = {"file": uploaded_file.getvalue()}

            response = requests.post(f"{BACKEND_URL}/predict", files=files)

            result = response.json()

            prediction = result["prediction"]
            probability = result["probability"]

            st.subheader("🧠 AI Prediction")

            if prediction == "PNEUMONIA":
                st.error(f"PNEUMONIA detected ({probability:.3f})")
            else:
                st.success(f"NORMAL ({probability:.3f})")

            # GradCAM decode
            img_data = base64.b64decode(result["gradcam_image"])
            gradcam_img = Image.open(io.BytesIO(img_data))

            st.subheader("🔍 Grad-CAM Explanation")

            image_comparison(
                img1=image,
                img2=gradcam_img,
                label1="Original",
                label2="Grad-CAM",
                width=700
            )