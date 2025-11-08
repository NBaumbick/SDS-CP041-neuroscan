# app.py
import io
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras

MODEL_PATH  = Path(__file__).resolve().parent / "models" / "best_model.keras"  # or final_model.keras
IMG_SIZE    = (240, 240)
THRESHOLD   = 0.5
CLASS_NAMES = ["No Tumor", "Tumor"]

st.set_page_config(page_title="🧠 Brain Tumor Scan Predictor", page_icon="🧠")
st.title("🧠 Brain Tumor Scan Predictor")
st.caption(f"Loading model: {MODEL_PATH.name}")

@st.cache_resource(show_spinner=True)
def load_full_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    return keras.models.load_model(str(MODEL_PATH))

def preprocess(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[None, ...]  # (1, 240, 240, 3)

uploaded = st.file_uploader("Upload JPG/PNG scan", type=["jpg", "jpeg", "png"])
if uploaded:
    pil = Image.open(io.BytesIO(uploaded.read()))
    st.image(pil, caption="Uploaded image", use_column_width=True)

    if st.button("Run Prediction", use_container_width=True):
        try:
            with st.spinner("Predicting…"):
                model = load_full_model()
                x = preprocess(pil)
                p = float(model.predict(x, verbose=0)[0, 0])  # binary sigmoid
                label = CLASS_NAMES[1] if p >= THRESHOLD else CLASS_NAMES[0]
            st.subheader(f"Prediction: {label}")
            st.write(f"Tumor probability: **{p:.4f}** (threshold {THRESHOLD})")
        except Exception as e:
            st.exception(e)
else:
    st.info("Upload a scan to begin.")
