import streamlit as st
import numpy as np
import cv2
import joblib
import pandas as pd
import plotly.graph_objects as go
from skimage.feature import hog, graycomatrix, graycoprops
from PIL import Image

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="TB Detector | Chest X-ray Screening",
    page_icon="🫁",
    layout="wide"
)

# ---------------------------------------------------------
# DARK THEME CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .main-title {
        font-size: 2.4rem; font-weight: 700; text-align: center;
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 10px 0px;
    }
    .subtitle { text-align: center; color: #9ca3af; font-size: 1.05rem; margin-bottom: 25px; }
    .model-card {
        padding: 14px; border-radius: 12px; text-align: center; margin-bottom: 10px;
    }
    .tb-positive { background-color: rgba(255, 75, 75, 0.12); border: 1px solid #ff4b4b; }
    .tb-negative { background-color: rgba(0, 200, 120, 0.12); border: 1px solid #00c878; }
    .summary-card {
        padding: 22px; border-radius: 14px; text-align: center; margin: 15px 0px;
        background-color: rgba(0, 114, 255, 0.10); border: 1px solid #0072ff;
    }
    .warning-note {
        background-color: rgba(255, 193, 7, 0.10); border: 1px solid #ffc107;
        padding: 12px; border-radius: 10px; font-size: 0.85rem; margin-top: 10px;
    }
    .footer-note { text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 40px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL REGISTRY
# Note: KNN is intentionally included for transparency, but it is flagged
# because it has very low precision on this dataset (over-predicts TB).
# ---------------------------------------------------------
MODEL_FILES = {
    "SVM (RBF)": "model_svm.pkl",
    "Logistic Regression": "model_logistic_regression.pkl",
    "Random Forest": "model_random_forest.pkl",
    "Decision Tree": "model_decision_tree.pkl",
}
# Note: KNN is intentionally excluded from live predictions — its saved model file
# is ~347MB (KNN stores the full training set), too large for GitHub/Render deployment.
# Its known test-set performance is still shown in the sidebar table for transparency.

IMG_SIZE = (128, 128)

# ---------------------------------------------------------
# LOAD MODELS + SCALER (cached so they load only once)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    models = {name: joblib.load(fname) for name, fname in MODEL_FILES.items()}
    scaler = joblib.load("scaler.pkl")
    return models, scaler

models, scaler = load_artifacts()

# ---------------------------------------------------------
# FEATURE EXTRACTION (must match training pipeline exactly)
# ---------------------------------------------------------
def extract_hog_features(img_gray):
    features, _ = hog(
        img_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
        block_norm='L2-Hys', visualize=True, feature_vector=True
    )
    return features

def extract_glcm_features(img_gray):
    img_uint8 = (img_gray * 255).astype(np.uint8) if img_gray.max() <= 1.0 else img_gray.astype(np.uint8)
    glcm = graycomatrix(
        img_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True
    )
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for prop in props:
        feats.extend(graycoprops(glcm, prop).flatten())
    return np.array(feats)

def extract_combined_features(img_array):
    img = cv2.resize(img_array, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    hog_feats = extract_hog_features(img)
    glcm_feats = extract_glcm_features(img)
    return np.concatenate([hog_feats, glcm_feats])

def predict_all_models(img_array):
    feats = extract_combined_features(img_array)
    feats_scaled = scaler.transform(feats.reshape(1, -1))

    predictions = {}
    for name, model in models.items():
        pred = model.predict(feats_scaled)[0]
        proba = model.predict_proba(feats_scaled)[0][1] if hasattr(model, "predict_proba") else float(pred)
        predictions[name] = {"pred": int(pred), "proba": float(proba)}
    return predictions

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown('<div class="main-title">🫁 TB Chest X-ray Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a chest X-ray — see predictions from 4 classical ML models side-by-side (HOG + GLCM features)</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1.4])

with col1:
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L")
        st.image(image, caption="Uploaded X-ray", width='stretch')

with col2:
    if uploaded_file:
        img_array = np.array(image)
        with st.spinner("Running all 5 models..."):
            predictions = predict_all_models(img_array)

        tb_votes = sum(1 for p in predictions.values() if p["pred"] == 1)
        total_models = len(predictions)
        majority_threshold = total_models // 2 + 1  # strict majority

        # ---- Majority vote summary ----
        if tb_votes >= majority_threshold:
            st.markdown(f"""
            <div class="summary-card">
                <h3>⚠️ {tb_votes} out of {total_models} models predict Tuberculosis</h3>
                <p>Majority signal suggests possible TB — consult a radiologist for confirmation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="summary-card">
                <h3>✅ {total_models - tb_votes} out of {total_models} models predict Normal</h3>
                <p>Majority signal suggests Normal — but always confirm with a medical professional.</p>
            </div>
            """, unsafe_allow_html=True)

        # ---- Individual model cards ----
        st.markdown("#### Individual Model Predictions")
        cols = st.columns(4)
        display_order = ["SVM (RBF)", "Logistic Regression", "Random Forest", "Decision Tree"]

        for i, name in enumerate(display_order):
            p = predictions[name]
            card_class = "tb-positive" if p["pred"] == 1 else "tb-negative"
            label = "TB" if p["pred"] == 1 else "Normal"
            icon = "⚠️" if p["pred"] == 1 else "✅"
            with cols[i]:
                st.markdown(f"""
                <div class="model-card {card_class}">
                    <b>{name}</b><br>
                    {icon} {label}<br>
                    <span style="font-size:0.8rem;">{p['proba']*100:.1f}% TB prob.</span>
                </div>
                """, unsafe_allow_html=True)

        # ---- Bar chart comparison ----
        st.markdown("#### TB Probability by Model")
        fig = go.Figure(go.Bar(
            x=list(predictions.keys()),
            y=[p["proba"] * 100 for p in predictions.values()],
            marker_color=["#ff4b4b" if p["pred"] == 1 else "#00c878" for p in predictions.values()],
            text=[f"{p['proba']*100:.1f}%" for p in predictions.values()],
            textposition='outside'
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#fafafa"}, height=350,
            yaxis=dict(title="TB Probability (%)", range=[0, 105], gridcolor="#333"),
            xaxis=dict(title="")
        )
        st.plotly_chart(fig, width='stretch')

    else:
        st.info("👈 Upload an X-ray image to see predictions from all 5 models")

# ---------------------------------------------------------
# SIDEBAR — About / Model Performance Reference
# ---------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About This Model")
    st.write("""
    This tool runs **4 classical ML algorithms live** (a 5th, KNN, was trained
    too — see its results in the table below) on the Kaggle
    *Tuberculosis (TB) Chest X-ray Database*, using HOG + GLCM texture features.

    **Class imbalance handling:** SMOTE oversampling + balanced class weights
    """)

    st.markdown("---")
    st.subheader("📊 Model Performance (Test Set)")
    try:
        perf_df = pd.read_csv("model_comparison.csv")
        st.dataframe(
            perf_df.style.format({
                "Accuracy": "{:.1%}", "Precision": "{:.1%}",
                "Recall": "{:.1%}", "F1-Score": "{:.1%}", "ROC-AUC": "{:.3f}"
            }),
            width='stretch', hide_index=True
        )
    except FileNotFoundError:
        st.caption("model_comparison.csv not found — run the training pipeline first.")

    st.markdown("""
    <div class="warning-note">
    ⚠️ <b>Note on KNN:</b> KNN was trained and evaluated (see table above — 100% recall
    but very low precision, ~29%), but is <b>not included in live predictions</b> here:
    its saved model file is ~347MB (KNN stores the entire training set), too large
    to deploy. It also over-predicts TB for almost every image, so SVM / Logistic
    Regression / Random Forest remain the more trustworthy choices anyway.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ Educational/portfolio project — not a certified diagnostic tool. Always consult a radiologist/physician.")
    st.caption("Built by Gunika Tanwar | Data Science Portfolio Project")

st.markdown('<div class="footer-note">TB Detector · 4-Model Classical ML Comparison (KNN evaluated, not deployed live) · Deployed on Render</div>', unsafe_allow_html=True)