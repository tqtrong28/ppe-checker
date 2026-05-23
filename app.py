"""Streamlit UI for SH17 PPE Compliance Checker."""

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from compliance import check_compliance
from detector import Detector


@st.cache_resource
def load_detector() -> Detector:
    return Detector(config.WEIGHTS_PATH)


def annotate(image_rgb: np.ndarray, detections, compliance_results) -> np.ndarray:
    """Draw boxes. Persons get index labels; PPE-found vs PPE-orphan colored differently."""
    img = image_rgb.copy()
    persons_iter = iter(compliance_results)
    persons_drawn = 0
    accepted_ppe_ids = set()
    for pc in compliance_results:
        for group_dets in pc.found_ppe.values():
            for d in group_dets:
                accepted_ppe_ids.add(id(d))

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        if det.class_id == config.PERSON_CLASS_ID:
            color = config.GROUP_COLORS_BGR["person"]
            label = f"Person #{persons_drawn + 1} {det.confidence:.2f}"
            persons_drawn += 1
        elif det.class_id in config.PERSON_GROUP_CLASS_IDS:
            color = config.GROUP_COLORS_BGR["head_face"]
            label = f"{det.class_name} {det.confidence:.2f}"
        elif det.class_name in config.PPE_TO_GROUP:
            if id(det) in accepted_ppe_ids:
                color = config.GROUP_COLORS_BGR["ppe_found"]
            else:
                color = config.GROUP_COLORS_BGR["ppe_orphan"]
            label = f"{det.class_name} {det.confidence:.2f}"
        else:
            color = config.GROUP_COLORS_BGR["other"]
            label = f"{det.class_name} {det.confidence:.2f}"
        rgb = color[::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
        cv2.putText(
            img, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 1, cv2.LINE_AA,
        )
    return img


def resize_for_display(image_rgb: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    if w <= max_width:
        return image_rgb
    new_w = max_width
    new_h = int(h * new_w / w)
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def build_compliance_df(compliance_results, required_groups: list[str]) -> pd.DataFrame:
    rows = []
    for pc in compliance_results:
        row = {"Person": f"#{pc.person_idx + 1}"}
        for g in required_groups:
            items = pc.found_ppe.get(g, [])
            if items:
                names = ", ".join(sorted({d.class_name for d in items}))
                row[g] = f"OK ({names})"
            else:
                row[g] = "MISSING"
        row["Status"] = "Compliant" if pc.is_compliant else "Violation"
        rows.append(row)
    return pd.DataFrame(rows)


st.set_page_config(page_title="SH17 PPE Compliance Checker", layout="wide")
st.title("SH17 PPE Compliance Checker")

st.sidebar.header("Detection Settings")
conf_person = st.sidebar.slider(
    "Person/Head confidence",
    min_value=0.05, max_value=0.95, value=config.DEFAULT_CONF_PERSON, step=0.05,
)
conf_ppe = st.sidebar.slider(
    "PPE confidence",
    min_value=0.05, max_value=0.95, value=config.DEFAULT_CONF_PPE, step=0.05,
)

st.sidebar.header("Required PPE")
req_head = st.sidebar.checkbox("Head (Helmet)", value=True)
req_body = st.sidebar.checkbox("Body (Vest/Suit)", value=True)
req_hand = st.sidebar.checkbox("Hand (Gloves)", value=False)
req_foot = st.sidebar.checkbox("Foot (Shoes)", value=False)

required_groups: list[str] = []
if req_head:
    required_groups.append("HEAD")
if req_body:
    required_groups.append("BODY")
if req_hand:
    required_groups.append("HAND")
if req_foot:
    required_groups.append("FOOT")

with st.sidebar.expander("About"):
    st.markdown(
        "**Dataset:** SH17 (8,099 images, 17 PPE-related classes; "
        "[GitHub](https://github.com/ahmadmughees/SH17dataset), CC BY-NC-SA 4.0).\n\n"
        "**Model:** YOLOv9-s pre-trained on SH17 (paper mAP50 = 65.3%).\n\n"
        "**Limitations:** Rare PPE classes (helmet, safety-vest, face-guard, "
        "safety-suit, medical-suit) have lower recall. Lower the PPE confidence "
        "threshold if items are visibly present but undetected.\n\n"
        "**Spatial association** is approximate. A helmet is attributed to a person "
        "when its bounding box is at least 70% inside the person's bounding box. "
        "Occlusion or close stacks of people may cause incorrect attribution. "
        "This app is for educational/demo purposes, not safety-critical use."
    )

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
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
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
    detections = detector.predict(image_np, conf_person, conf_ppe)

compliance_results = check_compliance(
    detections,
    required_groups=required_groups,
    containment_threshold=config.CONTAINMENT_THRESHOLD,
)

annotated = annotate(image_np, detections, compliance_results)
annotated_display = resize_for_display(annotated, config.MAX_DISPLAY_WIDTH)

n_persons = len(compliance_results)
n_compliant = sum(1 for pc in compliance_results if pc.is_compliant)
n_violations = n_persons - n_compliant

m1, m2, m3 = st.columns(3)
m1.metric("Total persons", n_persons)
m2.metric("Compliant", n_compliant)
m3.metric("Violations", n_violations)

col_img, col_table = st.columns([2, 1])
with col_img:
    st.image(annotated_display, caption=f"{len(detections)} total detections")
with col_table:
    if not required_groups:
        st.info("Select at least one required PPE group in the sidebar.")
    elif n_persons == 0:
        st.warning(
            "No person detected. Try lowering the Person/Head confidence threshold."
        )
    else:
        df = build_compliance_df(compliance_results, required_groups)
        st.dataframe(df, hide_index=True, use_container_width=True)
