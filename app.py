import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os

st.set_page_config(page_title="YOLO Object Detection")
st.title("YOLOv11 Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", ["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
        image.save(f.name)
        temp_path = f.name

    results = model(temp_path)
    annotated = results[0].plot()

    st.image(annotated, caption="Detected Objects", use_column_width=True)

    os.remove(temp_path)
