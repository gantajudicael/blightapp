import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌿 Excess Green (ExG) Index Visualization for Potato Leaves")
st.markdown("Upload a leaf image to calculate and visualize the Excess Green Index (ExG).")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert to RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original Image", use_column_width=True)

    # Extract channels
    R = image_rgb[:, :, 0].astype(np.float32)
    G = image_rgb[:, :, 1].astype(np.float32)
    B = image_rgb[:, :, 2].astype(np.float32)

    # Calculate ExG
    exg = 2 * G - R - B
    mean_exg = np.mean(exg)

    # Normalize for visualization
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Display index map
    st.subheader(f"🧪 Mean ExG: {mean_exg:.4f}")
    st.image(exg_norm, caption="ExG Visualization", use_column_width=True, clamp=True)

    # Threshold slider
    threshold_exg = st.slider("Threshold for ExG (infection suspicion if < value)", min_value=-100.0, max_value=100.0, value=50.0, step=0.1)

    # ExG Mask
    exg_mask = (exg < threshold_exg).astype(np.uint8) * 255

    st.subheader("⚠️ Suspected Infection Regions (Mask)")
    st.image(exg_mask, caption="ExG Infection Mask", use_column_width=True, clamp=True)