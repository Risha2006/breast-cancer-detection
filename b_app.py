
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BUNDLE = Path("artifacts/breast_cancer_lr.joblib")
assets = joblib.load(BUNDLE)
model = assets["model"]
feature_names = assets["feature_names"]
target_names = assets["target_names"]

st.set_page_config(page_title="Breast Cancer Classifier (Demo)")
st.title("Breast Cancer Detection (Demo)")
st.caption("Educational prototype using scikit-learn's Wisconsin dataset. Not for clinical use.")

with st.expander("How to use"):
    st.write("""
    - Enter values for the 30 features below (they match the Wisconsin dataset features).
    - Click **Predict** to get the probability of malignant.
    """)

# Sidebar quick-fill (mean values)
st.sidebar.header("Quick Fill")
if st.sidebar.button("Use mean feature values"):
    st.session_state["inputs"] = {f: 0.0 for f in feature_names}

st.header("Inputs")
cols = st.columns(3)
inputs = {}
for i, f in enumerate(feature_names):
    with cols[i % 3]:
        # Float inputs; zero default; user can paste real values from dataset or measurements.
        val = st.number_input(f, value=float(0.0))
        inputs[f] = val

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=feature_names)
    proba_malignant = float(model.predict_proba(X)[:, 1])
    pred = int(proba_malignant >= 0.5)
    label = target_names[pred]
    st.subheader("Result")
    st.metric("Predicted probability (malignant)", f"{proba_malignant:.3f}")
    st.write(f"Predicted class: **{label}** (threshold 0.50)")
