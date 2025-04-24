import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Configurer la page Streamlit
st.set_page_config(layout="wide")
st.title("🌿 Excess Green (ExG) Index Visualization for Potato Leaves")
st.markdown("Upload a leaf image to calculate and visualize the Excess Green Index (ExG).")

# Télécharger l'image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Charger l'image
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convertir en RGB
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original Image", use_column_width=True)

    # --- Étape GrabCut pour suppression du fond ---
    # Créer un masque initial
    mask = np.zeros(image.shape[:2], np.uint8)

    # Définir les modèles de fond et premier plan (utilisés par GrabCut)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Définir le rectangle autour de la feuille (en essayant d'inclure uniquement la feuille)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)

    # Appliquer GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Créer le masque binaire final où 1=foreground (feuille), 0=background
    final_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Appliquer le masque à l'image originale
    result = image_rgb * final_mask[:, :, np.newaxis]

    # Convertir le résultat en BGR pour l'enregistrement ou l'affichage
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    st.image(result_bgr, caption="Image with Background Removed", use_column_width=True)

    # --- Calcul de l'Indice ExG ---
    # Extraire les canaux RGB
    R = result[:, :, 0].astype(np.float32)
    G = result[:, :, 1].astype(np.float32)
    B = result[:, :, 2].astype(np.float32)

    # Calculer l'ExG
    exg = 2 * G - R - B
    mean_exg = np.mean(exg)

    # Normaliser pour la visualisation
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Afficher la carte de l'indice
    st.subheader(f"🧪 Mean ExG: {mean_exg:.4f}")
    st.image(exg_norm, caption="ExG Visualization", use_column_width=True, clamp=True)

    # Slider pour le seuil de l'ExG
    threshold_exg = st.slider("Threshold for ExG (infection suspicion if < value)", min_value=-100.0, max_value=100.0, value=0.05, step=0.1)

    # Masque ExG
    exg_mask = (exg < threshold_exg).astype(np.uint8) * 255

    st.subheader("⚠️ Suspected Infection Regions (Mask)")
    st.image(exg_mask, caption="ExG Infection Mask", use_column_width=True, clamp=True)
