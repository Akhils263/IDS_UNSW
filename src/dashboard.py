import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess_upload import preprocess_uploaded

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# --- Page config ---
st.set_page_config(page_title="IDS Monitoring Dashboard", layout="wide")
st.title("🛡️ Intrusion Detection & Monitoring Dashboard")

# --- Load models ---
@st.cache_resource
def load_models():
    binary_model = joblib.load(MODELS_DIR / "rf_binary.pkl")
    attack_model = joblib.load(MODELS_DIR / "rf_attack_category.pkl")
    return binary_model, attack_model

model, attack_model = load_models()

# --- File uploader ---
st.info("Upload a CSV of raw network traffic features (no label column needed).")
uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_data.head())

    # --- Preprocess ---
    try:
        processed_data = preprocess_uploaded(raw_data)
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # --- Predict ---
    try:
        predictions   = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1]
        attack_preds  = attack_model.predict(processed_data)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    results = raw_data.copy()
    results["Prediction"]         = predictions
    results["Attack_Probability"] = probabilities
    results["Risk_Score"]         = (probabilities * 100).round(2)
    results["Status"]             = results["Prediction"].map({0: "Normal", 1: "Attack"})
    results["Attack_Type"]        = attack_preds
    # For normal traffic, set attack type to "Normal"
    results.loc[results["Prediction"] == 0, "Attack_Type"] = "Normal"

    # --- Summary metrics ---
    total   = len(results)
    attacks = int(results["Prediction"].sum())
    normal  = total - attacks

    st.subheader("Detection Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records",    total)
    col2.metric("Attacks Detected", attacks)
    col3.metric("Normal Traffic",   normal)

    # --- Charts side by side ---
    chart_col1, chart_col2, _ = st.columns(3)

    with chart_col1:
        st.subheader("Attack Distribution")
        fig, ax = plt.subplots()
        ax.pie([normal, attacks], labels=["Normal", "Attack"], autopct="%1.1f%%")
        st.pyplot(fig)

    with chart_col2:
        st.subheader("Risk Score Distribution")
        fig2, ax2 = plt.subplots()
        ax2.hist(results["Risk_Score"], bins=20)
        ax2.set_xlabel("Risk Score")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)

    # --- Flagged attacks table ---
    st.subheader("Detected Attacks")
    st.dataframe(results[results["Prediction"] == 1][["Status", "Attack_Type", "Risk_Score", "Attack_Probability"]])

    # --- Attack type breakdown ---
    if attacks > 0:
        st.subheader("Attack Type Breakdown")
        breakdown = results[results["Prediction"] == 1]["Attack_Type"].value_counts().reset_index()
        breakdown.columns = ["Attack Type", "Count"]
        st.dataframe(breakdown)

    # --- Download ---
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download Results", csv, "ids_results.csv", "text/csv")