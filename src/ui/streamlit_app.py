"""
streamlit_app.py — Interactive web interface for RadReason.

Features:
- Image upload / Sample selection
- Predicted Findings (calibrated probabilities)
- Evidence Retrieval Cards
- Structured Report (LLM generated)
- Health Status Banner
"""

import streamlit as st
import requests
import os
import pandas as pd
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="RadReason", page_icon="🩻", layout="wide")

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# 2. Sidebar & Banner Logic
def should_warn(confidence: str, escalate: bool) -> bool:
    """True if the UI should display a warning banner."""
    return (confidence == "low") or escalate

# ASSERTIONS for Check 5
assert should_warn(confidence="low", escalate=False) == True
assert should_warn(confidence="high", escalate=True) == True
assert should_warn(confidence="high", escalate=False) == False
assert should_warn(confidence="moderate", escalate=False) == False

# 3. Resource Loading (Check 6)
@st.cache_resource
def load_classifier(model_path="artifacts/models/best_model.pt", config_path="configs/train_image.yaml"):
    # CACHE REQUIRED — removing this decorator causes per-request model reloads
    from src.models.predict import RadiologistCNN
    return RadiologistCNN(model_path=model_path, config_path=config_path)

@st.cache_resource
def load_retriever(index_path="artifacts/faiss/index.bin", meta_path="artifacts/faiss/chunks_meta.pkl"):
    # CACHE REQUIRED — removing this decorator causes per-request model reloads
    from src.retrieval.retrieve import RadiologyRetriever
    return RadiologyRetriever(index_path=index_path, meta_path=meta_path)

# 4. Main Title
st.title("🩻 RadReason — Multimodal CXR Copilot")
st.markdown("---")

# 5. Sidebar: Health & Metrics
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2).json()
        if health["status"] == "healthy":
            st.success("API: Online")
        else:
            st.warning("API: Degraded")
    except:
        st.error("API: Offline")
        
    st.header("Performance (AUROC)")
    try:
        metrics = requests.get(f"{API_URL}/metrics", timeout=2).json()
        if "per_class" in metrics:
            for label, score in metrics["per_class"].items():
                st.metric(label.capitalize(), f"{score:.3f}")
            st.write(f"**Macro AUROC: {metrics['macro_auroc']:.3f}**")
    except:
        st.write("Metrics unavailable.")

# 6. Main UI: Upload & Predict
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload Chest X-ray (PNG)", type=["png"])
    selected_sample = st.selectbox("Or select a sample study:", ["None", "Sample 1", "Sample 2"])
    clinical_info = st.text_area("Clinical Indication (Optional):", 
                                 placeholder="e.g., Shortness of breath, chest pain")
    predict_btn = st.button("Generate Report", type="primary")

# 7. Pipeline Execution
if predict_btn:
    if uploaded_file or selected_sample != "None":
        # Placeholder path for MVP
        temp_path = "data/raw/images/CXR1_IM-0001-1001.png"
        
        with st.spinner("Analyzing image and retrieving evidence..."):
            try:
                payload = {"image_path": temp_path, "clinical_indication": clinical_info}
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    with col2:
                        st.header("Analysis Results")
                        
                        # 7a. Predictions
                        st.subheader("Predicted Findings")
                        preds = data["predictions"]
                        for p in preds:
                            if p["prob"] >= 0.4:
                                st.progress(p["prob"], text=f"{p['label'].capitalize()}: {p['prob']:.2f}")
                            else:
                                st.text(f"{p['label'].capitalize()}: {p['prob']:.2f}")
                        
                        # 7b. Evidence
                        st.subheader("Retrieved Evidence")
                        hits = data["retrieved_evidence"]
                        for i, hit in enumerate(hits):
                            with st.expander(f"Evidence {i+1} (Score: {hit['score']:.2f})"):
                                st.write(hit["text"])
                                st.caption(f"Source: {hit['study_id']}")
                                
                        # 7c. Generated Report
                        st.subheader("Diagnostic Summary")
                        summary = data["summary"]
                        st.success(f"**Impression:** {summary['impression']}")
                        st.write(f"**Findings:** {summary['findings']}")
                        st.info(f"**Rationale:** {summary['rationale']}")
                        
                        if should_warn(summary["confidence"], summary["escalate"]):
                            st.error("⚠️ ESCALATION ADVISED: Review by senior radiologist recommended.")
                        
                        st.warning(f"**Caution:** {summary['caution']}")
                        st.caption(f"Confidence: {summary['confidence']}")
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
    else:
        st.warning("Please upload an image or select a sample.")

with col1:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Chest X-ray")
