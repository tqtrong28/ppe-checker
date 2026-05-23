"""Streamlit UI for SH17 PPE Compliance Checker."""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import config
from detector import Detector


@st.cache_resource
def load_detector() -> Detector:
    return Detector(config.WEIGHTS_PATH)


def annotate(image_rgb: np.ndarray, detections) -> np.ndarray:
    """Draw bounding boxes for every detection. Color by class group."""
    img = image_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if det.class_id == config.PERSON_CLASS_ID:
            color = config.GROUP_COLORS_BGR["person"]
        elif det.class_id in config.PERSON_GROUP_CLASS_IDS:
            color = config.GROUP_COLORS_BGR["head_face"]
        elif det.class_name in config.PPE_TO_GROUP:
            color = config.GROUP_COLORS_BGR["ppe_found"]
        else:
            color = config.GROUP_COLORS_BGR["other"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color[::-1], 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            img, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[::-1], 1, cv2.LINE_AA,
        )
    return img


def resize_for_display(image_rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    new_w = max_width
    new_h = int(h * new_w / w)
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


st.set_page_config(page_title="SH17 PPE Compliance Checker", layout="wide")
st.title("SH17 PPE Compliance Checker")

uploaded = st.sidebar.file_uploader(
    "Upload workplace image", type=["jpg", "jpeg", "png"]
)

try:
    detector = load_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if uploaded is None:
    st.info("Upload a JPG or PNG image to begin. Use the sidebar.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

image_np = np.array(image)
h, w = image_np.shape[:2]
if w > config.MAX_INFERENCE_WIDTH:
    new_w = config.MAX_INFERENCE_WIDTH
    new_h = int(h * new_w / w)
    image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    st.info(f"Image was resized for inference: {w}x{h} -> {new_w}x{new_h}")

with st.spinner("Detecting..."):
    detections = detector.predict(
        image_np,
        conf_person=config.DEFAULT_CONF_PERSON,
        conf_ppe=config.DEFAULT_CONF_PPE,
    )

annotated = annotate(image_np, detections)
annotated_display = resize_for_display(annotated, config.MAX_DISPLAY_WIDTH)

st.image(annotated_display, caption=f"{len(detections)} detections")
