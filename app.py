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
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Freshness Quality Inspection",
    page_icon="🍎",
    layout="wide"
)

# ================= PROFESSIONAL ENTERPRISE UI =================
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at top, #020617, #020617);
    color: #e5e7eb;
    font-family: "Segoe UI", sans-serif;
}

.main-title {
    font-size: 3.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(90deg,#22c55e,#38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-size: 1.1rem;
}

.section-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #22c55e;
    margin: 2.2rem 0 1rem;
}

.card {
    background: rgba(15,23,42,0.75);
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid rgba(148,163,184,0.15);
    box-shadow: 0 25px 50px rgba(0,0,0,0.5);
    margin-bottom: 2rem;
}

.badge-low { color:#22c55e; font-weight:700; }
.badge-mid { color:#facc15; font-weight:700; }
.badge-high{ color:#ef4444; font-weight:700; }

.info-box {
    background: rgba(255,255,255,0.04);
    padding: 1rem;
    border-radius: 12px;
    border-left: 4px solid #38bdf8;
    margin-top: 0.7rem;
}

hr {
    border:none;
    height:1px;
    background: linear-gradient(90deg, transparent, #22c55e, transparent);
    margin:2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ================= MODEL =================
MODEL_URL = "https://github.com/SIVASEELAN333/Fruit-Freshness-AI/releases/download/v1.0/fruit_rotten_model.h5"
MODEL_PATH = "fruit_rotten_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading AI model..."):
            r = requests.get(MODEL_URL)
            r.raise_for_status()
            open(MODEL_PATH, "wb").write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# ================= YOLO =================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")

yolo_model = load_yolo()


# ================= SESSION =================
if "history" not in st.session_state:
    st.session_state.history = []


# ================= HEADER =================
st.markdown('<div class="main-title">🍎 AI-Based Freshness Quality Inspection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning · Object Detection · Explainable AI · Industry Grade System</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ================= SIDEBAR =================
with st.sidebar:
    st.header("⚙ Inspection Controls")
    use_gaussian = st.toggle("Noise Reduction (Gaussian)")
    use_yolo = st.toggle("YOLO Object Detection")
    show_gradcam = st.toggle("Explainable AI (Grad-CAM)")
    show_dashboard = st.toggle("📊 Visual Analytics Dashboard")
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "📤 Upload Inspection Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )


# ================= HELPERS =================
def estimate_shelf_life(p):
    if p < 30: return "5–7 Days"
    if p < 60: return "2–4 Days"
    return "0–1 Day"

def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No Conv Layer")

def make_gradcam(x, model, layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv, pred = grad_model(x)
        loss = tf.squeeze(pred) 
    grads = tape.gradient(loss, conv)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))
    conv = conv[0]
    heatmap = tf.reduce_sum(conv * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(hm, img):
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    hm = np.uint8(255 * hm)
    hm = cm.jet(hm)[:, :, :3]
    hm = np.uint8(hm * 255)
    return cv2.addWeighted(img, 0.6, hm, 0.4, 0)

def draw_yolo(img):
    out = img.copy()
    for r in yolo_model(img):
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), (34,197,94), 2)
    return out

def resize_for_display(img, max_width=640):
    h, w, _ = img.shape
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def ai_interpretation_message(p):
    if p < 10:
        return ("#22c55e","🟢 Excellent Freshness",
                "Surface appears extremely fresh with uniform color and firm texture. "
                "No degradation patterns detected.")
    elif p < 20:
        return ("#4ade80","🟢 Very Fresh",
                "Color tone remains natural with minimal variation. "
                "Texture integrity is well preserved.")
    elif p < 30:
        return ("#a3e635","🟡 Slight Aging",
                "Very minor surface variation observed. "
                "Early aging signs may be present but quality remains good.")
    elif p < 40:
        return ("#fde047","🟡 Early Spoilage",
                "Small discoloration patches detected. "
                "Initial freshness degradation has begun.")
    elif p < 50:
        return ("#facc15","🟠 Quality Declining",
                "Visible texture inconsistency observed. "
                "Product freshness is reducing gradually.")
    elif p < 60:
        return ("#fb923c","🟠 Moderate Spoilage",
                "Surface softness and color distortion detected. "
                "Moderate spoilage indicators present.")
    elif p < 70:
        return ("#f87171","🔴 High Spoilage",
                "Noticeable decay regions identified. "
                "Consumption should be avoided soon.")
    elif p < 80:
        return ("#ef4444","🔴 Severe Spoilage",
                "Strong decay patterns including dark regions detected. "
                "Product quality is critically low.")
    elif p < 90:
        return ("#b91c1c","🔴 Near Complete Decay",
                "Extensive surface breakdown and heavy discoloration observed. "
                "Item is nearly rotten.")
    else:
        return ("#7f1d1d","⚫ Unsafe for Consumption",
                "Extreme spoilage patterns detected. "
                "Product is unsafe and must be discarded immediately.")


# ================= MAIN =================
if uploaded_files:

    st.markdown('<div class="section-title">📊 Inspection Summary</div>', unsafe_allow_html=True)

    total = len(uploaded_files)
    fresh = rotten = high = 0

    for f in uploaded_files:
        img = np.array(Image.open(f).convert("RGB"))
        x = cv2.resize(img, (224,224)) / 255.0
        p = float(model.predict(np.expand_dims(x,0), verbose=0)[0][0]) * 100
        fresh += p < 50
        rotten += p >= 50
        high += p >= 70

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Samples", total)
    c2.metric("Fresh Items", fresh)
    c3.metric("Rotten Items", rotten)
    c4.metric("High Risk", high)

    st.markdown("<hr>", unsafe_allow_html=True)

    batch = []

    for f in uploaded_files:

        st.markdown('<div class="card">', unsafe_allow_html=True)

        img = np.array(Image.open(f).convert("RGB"))
        processed = cv2.GaussianBlur(img,(5,5),0) if use_gaussian else img

        processed = resize_for_display(processed)

        processed = draw_yolo(processed) if use_yolo else processed


        col1,col2 = st.columns([1.1,1])

        with col1:
            st.image(img, "Original Image", use_container_width=True)
            st.image(processed, "Processed Image")

        with col2:
            x = cv2.resize(processed,(224,224)) / 255.0
            p = float(model.predict(np.expand_dims(x,0), verbose=0)[0][0]) * 100
            progress = float(max(p,100-p)) / 100

            st.subheader("🧠 AI Analysis Report")
            st.metric("Freshness (%)", f"{100-p:.2f}")
            st.metric("Rotten Probability (%)", f"{p:.2f}")
            st.progress(progress)

            shelf = estimate_shelf_life(p)
            risk = "High" if p>=70 else "Medium" if p>=40 else "Low"
            badge = "badge-high" if risk=="High" else "badge-mid" if risk=="Medium" else "badge-low"

            st.markdown(f"**Risk Level:** <span class='{badge}'>{risk}</span>", unsafe_allow_html=True)
            st.markdown(f"**Estimated Shelf Life:** {shelf}")

            color,title,message = ai_interpretation_message(p)

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-left:6px solid {color};
            padding:16px;border-radius:12px;margin-top:12px;">
            <b style="color:{color};">{title}</b><br>{message}
            </div>
            """, unsafe_allow_html=True)

        if show_gradcam:
            hm = make_gradcam(np.expand_dims(x,0), model, get_last_conv_layer(model))
            st.image(overlay_heatmap(hm, processed), caption="Grad-CAM", use_container_width=True)

        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": f.name,
            "Fresh %": round(100-p,2),
            "Rotten %": round(p,2),
            "Risk": risk,
            "Shelf Life": shelf
        }

        st.session_state.history.append(record)
        batch.append(record)

        st.markdown("</div>", unsafe_allow_html=True)


    # ================= BATCH REPORT =================
    st.markdown('<div class="section-title">📦 Batch Inspection Report</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(batch), use_container_width=True)


    # ================= VISUAL ANALYTICS DASHBOARD =================
    if show_dashboard and batch:

        st.markdown('<div class="section-title">📊 Visual Analytics Dashboard</div>', unsafe_allow_html=True)

        df_batch = pd.DataFrame(batch)
        colA, colB = st.columns(2)

        with colA:
            st.subheader("🍏 Fresh vs Rotten Distribution")
            fresh_count = (df_batch["Risk"] == "Low").sum()
            rotten_count = len(df_batch) - fresh_count

            fig1, ax1 = plt.subplots()
            ax1.pie([fresh_count, rotten_count],
                    labels=["Fresh","Rotten"],
                    autopct="%1.1f%%",
                    startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

        with colB:
            st.subheader("🧪 Spoilage Severity Distribution")
            spoilage = df_batch["Risk"].value_counts()

            fig2, ax2 = plt.subplots()
            ax2.bar(spoilage.index, spoilage.values)
            ax2.set_xlabel("Risk Level")
            ax2.set_ylabel("Samples")
            ax2.set_title("Spoilage Severity")
            st.pyplot(fig2)


# ================= HISTORY =================
if st.session_state.history:
    st.markdown('<div class="section-title">📈 Inspection History</div>', unsafe_allow_html=True)
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    c1,c2 = st.columns(2)
    c1.download_button("⬇ Download History", df.to_csv(index=False), "history.csv")
    if c2.button("🗑 Clear History"):
        st.session_state.history = []
        st.rerun()
