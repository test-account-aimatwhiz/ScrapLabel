import streamlit as st
import cv2
import numpy as np
import os

from sam2_engine import (
    load_sam2,
    generate_masks,
    save_crops,
    create_overlay
)

from utils import load_categories, add_category
from storage import save_to_db


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🧠 SAM2 Industrial Labeling System")


# =========================
# SIDEBAR CONTROLS (SAM2)
# =========================
st.sidebar.header("⚙️ SAM2 Parameters")

points_per_side = st.sidebar.slider("Points Per Side", 16, 128, 32, step=16)
points_per_batch = st.sidebar.slider("Points Per Batch", 16, 256, 128, step=16)
pred_iou_thresh = st.sidebar.slider("Pred IoU", 0.3, 0.9, 0.65, step=0.05)
stability_score_thresh = st.sidebar.slider("Stability Score", 0.5, 0.95, 0.75, step=0.05)
stability_score_offset = st.sidebar.slider("Stability Offset", 0.5, 1.5, 0.90, step=0.05)
crop_n_layers = st.sidebar.slider("Crop Layers", 0, 3, 1)
crop_overlap_ratio = st.sidebar.slider("Crop Overlap", 0.0, 0.8, 0.40, step=0.05)
min_mask_region_area = st.sidebar.slider("Min Mask Area", 10, 300, 50, step=10)
box_nms_thresh = st.sidebar.slider("NMS Threshold", 0.3, 0.9, 0.65, step=0.05)
use_m2m = st.sidebar.checkbox("Use M2M", value=True)


# =========================
# SESSION STATE
# =========================
if "image" not in st.session_state:
    st.session_state.image = None

if "image_name" not in st.session_state:
    st.session_state.image_name = None

if "crops" not in st.session_state:
    st.session_state.crops = []

if "index" not in st.session_state:
    st.session_state.index = 0


# =========================
# MODEL BUILDER (DYNAMIC)
# =========================
def build_generator():
    return load_sam2(
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        crop_n_layers=crop_n_layers,
        crop_overlap_ratio=crop_overlap_ratio,
        min_mask_region_area=min_mask_region_area,
        box_nms_thresh=box_nms_thresh,
        use_m2m=use_m2m
    )


mask_generator = build_generator()


# =========================
# UPLOAD IMAGE
# =========================
uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:

    st.session_state.image_name = uploaded.name

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.session_state.image = image


# =========================
# RUN SAM2
# =========================
if st.session_state.image is not None:

    if st.button("🚀 Run SAM2"):

        masks = generate_masks(mask_generator, st.session_state.image)
        crops = save_crops(st.session_state.image, masks)

        st.session_state.crops = crops
        st.session_state.index = 0

        overlay = create_overlay(st.session_state.image, masks)

        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        st.success(f"Detected {len(crops)} objects")


# =========================
# DISPLAY
# =========================
if st.session_state.image is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.image(st.session_state.image, caption="Original Image")

    with col2:
        if len(st.session_state.crops) > 0:
            st.image(
                st.session_state.crops[st.session_state.index],
                caption=f"{st.session_state.image_name}_object_{st.session_state.index}"
            )


# =========================
# LABELING SYSTEM
# =========================
if len(st.session_state.crops) > 0:

    st.subheader("🏷️ Label Objects")

    categories = load_categories()

    colA, colB = st.columns(2)

    with colA:
        label = st.selectbox("Select Label", categories)

    with colB:
        new_label = st.text_input("Add Category")

        if st.button("➕ Add"):
            if new_label.strip():
                add_category(new_label.strip())
                st.rerun()

    total = len(st.session_state.crops)
    idx = st.session_state.index

    st.info(f"📦 Total Objects: {total} | 🎯 {idx+1}/{total}")

    col1, col2, col3 = st.columns(3)

    # PREVIOUS
    with col1:
        if st.button("⬅️ Previous"):
            st.session_state.index = (idx - 1) % total
            st.rerun()

    # SAVE
    with col2:
        if st.button("💾 Save"):

            crop = st.session_state.crops[idx]
            object_id = f"{st.session_state.image_name}_object_{idx}"

            os.makedirs("database/crops", exist_ok=True)
            path = f"database/crops/{object_id}.png"

            cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

            save_to_db(
                st.session_state.image_name,
                object_id,
                label,
                path
            )

            st.success(f"Saved {object_id}")

    # NEXT
    with col3:
        if st.button("➡️ Next"):
            st.session_state.index = (idx + 1) % total
            st.rerun()


# =========================
# OVERLAY
# =========================
if os.path.exists("outputs/overlay.png"):
    st.image("outputs/overlay.png", caption="Segmentation Overlay")