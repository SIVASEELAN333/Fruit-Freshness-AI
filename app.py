import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime
from matplotlib import cm
import os
import requests

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Freshness Detection AI",
    page_icon="🍎",
    layout="wide"
)

# ---------------- Model Download ----------------
MODEL_URL = "https://github.com/SIVASEELAN333/Fruit-Freshness-AI/releases/download/v1.0/fruit_rotten_model.h5"
MODEL_PATH = "fruit_rotten_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI model (first-time setup)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------------- Session State ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- Header ----------------
st.markdown(
    """
    <h1 style="text-align:center;">🍎 AI-Based Fruit & Vegetable Freshness Detection</h1>
    <p style="text-align:center; color:#6c757d;">
    Explainable AI system with risk analysis, shelf-life estimation & batch inspection
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Inspection Settings")
    use_gaussian = st.toggle("Apply Gaussian Filter")
    show_gradcam = st.toggle("Show Explainable AI (Grad-CAM)")
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "📤 Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

# ---------------- Helper Functions ----------------
def estimate_shelf_life(rotten_prob):
    if rotten_prob < 30:
        return "5–7 days"
    elif rotten_prob < 60:
        return "2–4 days"
    else:
        return "0–1 day"

def image_quality_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    noise = np.std(gray)

    if blur < 50 or brightness < 40 or brightness > 210:
        return "Low"
    elif blur < 120 or noise > 25:
        return "Moderate"
    else:
        return "Good"

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No convolution layer found")

# ✅ FIXED GRAD-CAM (CLOUD SAFE)
def make_gradcam_heatmap(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.squeeze(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cm.jet(heatmap)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

# ---------------- Main Pipeline ----------------
if uploaded_files:

    st.markdown("## 📊 Inspection Summary")

    total = len(uploaded_files)
    fresh_count = 0
    rotten_count = 0
    high_risk = 0

    for f in uploaded_files:
        img = np.array(Image.open(f).convert("RGB"))
        x = cv2.resize(img, (224, 224)) / 255.0
        x = np.expand_dims(x, axis=0)
        prob = float(model.predict(x, verbose=0)[0][0]) * 100

        if prob >= 50:
            rotten_count += 1
        else:
            fresh_count += 1
        if prob >= 70:
            high_risk += 1

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total", total)
    c2.metric("🍏 Fresh", fresh_count)
    c3.metric("🧪 Rotten", rotten_count)
    c4.metric("⚠️ High Risk", high_risk)

    st.markdown("---")

    batch_results = []

    for uploaded_file in uploaded_files:
        st.markdown("## 🖼️ Image Analysis")

        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        if use_gaussian:
            img = cv2.GaussianBlur(img, (5, 5), 0)

        with col2:
            st.image(img, caption="Processed Image", use_container_width=True)

        quality = image_quality_metrics(img)
        if quality == "Good":
            st.success("📸 Image Quality: Good")
        elif quality == "Moderate":
            st.warning("📸 Image Quality: Moderate")
        else:
            st.error("📸 Image Quality: Low – re-upload recommended")

        x = cv2.resize(img, (224, 224)) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x, verbose=0)
        rotten_prob = float(pred[0][0]) * 100
        fresh_prob = 100 - rotten_prob

        col3, col4, col5 = st.columns(3)

        with col3:
            if rotten_prob > fresh_prob:
                confidence = rotten_prob
                st.error(f"🧪 Rotten\n\n{confidence:.2f}%")
            else:
                confidence = fresh_prob
                st.success(f"🍏 Fresh\n\n{confidence:.2f}%")

        with col4:
            st.metric("Fresh %", f"{fresh_prob:.2f}")
            st.metric("Rotten %", f"{rotten_prob:.2f}")

        with col5:
            st.progress(confidence / 100)
            shelf_life = estimate_shelf_life(rotten_prob)
            st.info(f"📆 Shelf Life\n\n{shelf_life}")

        if show_gradcam:
            last_conv = get_last_conv_layer(model)
            heatmap = make_gradcam_heatmap(x, model, last_conv)
            gradcam_img = overlay_heatmap(heatmap, img)
            st.subheader("🧠 Explainable AI (Grad-CAM)")
            st.image(gradcam_img, use_container_width=True)

        if rotten_prob >= 70:
            risk = "High"
            st.error("High Risk ⚠️ | Reject")
        elif rotten_prob >= 40:
            risk = "Medium"
            st.warning("Medium Risk ⚠️ | Monitor")
        else:
            risk = "Low"
            st.success("Low Risk ✅ | Safe")

        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": uploaded_file.name,
            "Fresh %": round(fresh_prob, 2),
            "Rotten %": round(rotten_prob, 2),
            "Risk": risk,
            "Shelf Life": shelf_life,
            "Image Quality": quality,
            "Gaussian Used": use_gaussian,
            "Grad-CAM": show_gradcam
        }

        st.session_state.history.append(record)
        batch_results.append(record)

        st.markdown("---")

    st.subheader("📦 Batch Inspection Summary")
    st.dataframe(pd.DataFrame(batch_results), use_container_width=True)

# ---------------- History ----------------
if st.session_state.history:
    st.subheader("📈 Inspection History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    col6, col7 = st.columns(2)
    with col6:
        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History", csv, "inspection_history.csv")
    with col7:
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()
