"""
=============================================================
  Brain Tumor Detection – Streamlit Web Application
  File: app.py
=============================================================
  HOW TO RUN LOCALLY:
    pip install streamlit tensorflow pillow opencv-python
    streamlit run app.py

  HOW TO DEPLOY (FREE):
    1. Push this file + brain_tumor_model.h5 to GitHub
    2. Go to https://streamlit.io/cloud
    3. Connect your GitHub repo and deploy
=============================================================
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time

#  Page Configuration 
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

#  Custom CSS 
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-tumor {
        background-color: #FDEDEC;
        border-left: 5px solid #E74C3C;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .result-notumor {
        background-color: #EAFAF1;
        border-left: 5px solid #2ECC71;
        padding: 1.2rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .disclaimer {
        font-size: 0.78rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
        font-style: italic;
    }
    .metric-box {
        background: #F4F6F7;
        border-radius: 8px;
        padding: 0.8rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

#  Load Model (cached) 
@st.cache_resource
def load_model():
    """Load the trained CNN model. Cached so it loads only once."""
    try:
        model = tf.keras.models.load_model("brain_tumor_model_mobilenet.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

IMG_SIZE = 224
THRESHOLD = 0.5

#  Preprocessing 
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE)) # PIL resize!
    img_array = np.array(img) / 255.0
    img_input = np.expand_dims(img_array, axis=0)
    return img, img_input

def get_confidence_bar(prob, label):
    """Return HTML for a styled confidence bar."""
    tumor_pct = prob * 100
    notumor_pct = (1 - prob) * 100
    return f"""
    <div style="margin: 1rem 0;">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#555; margin-bottom:4px;">
            <span>No Tumor</span><span>Tumor</span>
        </div>
        <div style="background:#eee; border-radius:10px; height:18px; position:relative;">
            <div style="background:#E74C3C; border-radius:10px; height:18px; width:{tumor_pct:.1f}%;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#888; margin-top:4px;">
            <span>{notumor_pct:.1f}%</span><span>{tumor_pct:.1f}%</span>
        </div>
    </div>
    """

#  Header 
st.markdown('<p class="main-title">🧠 Brain Tumor Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an MRI scan to detect the presence of a brain tumor using deep learning (CNN)</p>', unsafe_allow_html=True)

#  Sidebar 
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write("""
    This application uses a **Convolutional Neural Network (CNN)** trained on MRI brain scans to classify images as:
    - 🔴 **Tumor Detected**
    - 🟢 **No Tumor Found**
    """)

    st.subheader("📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "80.4%")
    col2.metric("F1-Score", "0.85")
    col1.metric("Precision", "80.0%")
    col2.metric("Recall", "90.3%")

    st.subheader("🛠️ Model Details")
    st.markdown("""
    - **Architecture**: MobileNetV2 (Transfer Learning)
    - **Framework**: TensorFlow / Keras
    - **Input Size**: 224 × 224 pixels
    - **Dataset**: Kaggle Brain MRI Dataset (253 images)
    - **Training**: 25 epochs, Adam optimizer
    """)

    

# ─── Main Upload Area ─────────────────────────────────────
st.subheader("📤 Upload MRI Image")
uploaded_file = st.file_uploader(
    "Choose an MRI image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI scan image for tumor detection"
)

if uploaded_file is not None:
    # 1. Load model
    model = load_model() 
    
    # 2. Preprocess image
    try:
        img_pil, img_input = preprocess_image(uploaded_file)
    except Exception:
        st.error("Invalid image file. Please upload a valid JPG or PNG.")
        st.stop()

    # 3. Display uploaded image
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("🖼️ Uploaded MRI Scan")
        st.image(img_pil, caption="Input MRI Image", use_column_width=True) 

    # Run prediction
    with col2:
        st.subheader("🔬 Analysis Result")
        with st.spinner("Analyzing MRI scan..."):
            time.sleep(0.8)  # Small delay for UX
            prob = float(model.predict(img_input, verbose=0)[0][0])

        is_tumor = prob >= THRESHOLD
        label = "TUMOR DETECTED" if is_tumor else "NO TUMOR FOUND"
        confidence = prob if is_tumor else (1 - prob)
        icon = "🔴" if is_tumor else "🟢"
        css_class = "result-tumor" if is_tumor else "result-notumor"
        color = "#E74C3C" if is_tumor else "#2ECC71"

        st.markdown(f"""
        <div class="{css_class}">
            <h2 style="color:{color}; margin:0;">{icon} {label}</h2>
            <p style="margin:0.5rem 0 0; color:#555;">
                Confidence: <strong>{confidence*100:.1f}%</strong><br>
                Raw probability (tumor): <strong>{prob:.4f}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(get_confidence_bar(prob, label), unsafe_allow_html=True)

        # Interpretation
        if is_tumor:
            st.warning("⚠️ The model detected possible signs of a tumor. Please consult a qualified radiologist for medical evaluation.")
        else:
            st.success("✅ No tumor detected in this scan. Regular check-ups are still recommended.")

    # Technical Details (expandable)
    with st.expander("🔧 Technical Details"):
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | Raw probability | `{prob:.6f}` |
        | Classification threshold | `{THRESHOLD}` |
        | Input image size | `{IMG_SIZE} × {IMG_SIZE} pixels` |
        | Model | Brain Tumor CNN v1.0 |
        | Framework | TensorFlow {tf.__version__} |
        """)

else:
    # Show example when no image uploaded
    st.info("👆 Upload an MRI brain scan image above to get started.")
    st.markdown("""
    **Sample images to try:**
    - Download test images from the [Kaggle dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
    - Use images from the `/yes` folder (tumor) or `/no` folder (no tumor)
    """)

# ─── Disclaimer ───────────────────────────────────────────
st.markdown('<p class="disclaimer">⚠️ DISCLAIMER: This application is a research prototype developed for educational purposes only. It is NOT a medical device and should NOT be used as a substitute for professional medical diagnosis. Always consult a qualified medical professional for health-related decisions.</p>', unsafe_allow_html=True)
