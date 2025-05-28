import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

model = tf.keras.models.load_model("best_unet.h5", compile=False)

st.title("COVID-19 CT Scan Segmentation App")

def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image).astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_mask(image):
    preprocessed = preprocess_image(image)
    pred = model.predict(preprocessed)[0]
    pred_mask = (pred > 0.5).astype(np.uint8)
    return pred_mask.squeeze()

uploaded_file = st.file_uploader("Upload a CT scan image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Segment"):
        pred_mask = predict_mask(image)

        st.subheader("Predicted Mask")
        st.image(pred_mask * 255, use_container_width=True, clamp=True)

        # Overlay the mask on the original image
        resized_image = image.resize((128, 128))
        original_np = np.array(resized_image)
        if original_np.ndim == 2:
            overlay = np.stack([original_np]*3, axis=-1)  # Convert grayscale to RGB
        else:
            overlay = original_np.copy()

        overlay[pred_mask == 1] = [255, 0, 0]  # Red mask overlay

        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)
