import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# App title
st.set_page_config(page_title="YOLO Object Detection", layout="wide")
st.title("🚀 YOLOv11 Object Detection App")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp_path = temp.name
        cv2.imwrite(temp_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    # Run YOLO
    results = model(temp_path)

    # Plot result
    annotated_img = results[0].plot()
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected Objects")
        st.image(annotated_img, use_column_width=True)

    # Cleanup
    os.remove(temp_path)
