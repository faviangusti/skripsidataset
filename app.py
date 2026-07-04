import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(
    page_title="PPE Detection",
    page_icon="🦺",
    layout="wide"
)

st.title("🦺 PPE Detection System")
st.write("Deteksi APD menggunakan model YOLOv8")

# Load model
model = YOLO("weights/best.pt")

confidence = st.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.05
)

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        results = model.predict(
            source=tmp.name,
            conf=confidence,
            save=False
        )

    result = results[0]

    plotted = result.plot()

    with col2:
        st.subheader("Hasil Deteksi")
        st.image(plotted, use_container_width=True)

    st.subheader("Ringkasan Deteksi")

    names = model.names

    counts = {}

    for box in result.boxes:
        cls = int(box.cls[0])
        label = names[cls]

        counts[label] = counts.get(label, 0) + 1

    if len(counts) == 0:
        st.info("Tidak ada objek yang terdeteksi.")
    else:
        for label, total in counts.items():
            st.write(f"**{label}** : {total}")