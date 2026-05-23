"""Streamlit UI for SH17 PPE Compliance Checker."""

import io

import numpy as np
import streamlit as st
from PIL import Image

import config


st.set_page_config(page_title="SH17 PPE Compliance Checker", layout="wide")
st.title("SH17 PPE Compliance Checker")

uploaded = st.sidebar.file_uploader(
    "Upload workplace image", type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    st.info("Upload a JPG or PNG image to begin. Use the sidebar.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

image_np = np.array(image)
h, w = image_np.shape[:2]

if w > config.MAX_INFERENCE_WIDTH:
    new_w = config.MAX_INFERENCE_WIDTH
    new_h = int(h * new_w / w)
    image = image.resize((new_w, new_h))
    image_np = np.array(image)
    st.info(f"Image was resized for inference: {w}x{h} -> {new_w}x{new_h}")

st.image(image_np, caption="Uploaded image", width=config.MAX_DISPLAY_WIDTH)
