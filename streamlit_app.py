import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import os

st.set_page_config(page_title="Bank Marketing ML Models", layout="wide")

st.title("📊 Bank Marketing Term Deposit Prediction")
st.write("Select a model and upload test data to generate predictions.")

# -----------------------------
# Load preprocessing pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("saved_models/preprocessing_pipeline.pkl")

pipeline = load_pipeline()

# -----------------------------
# Model selection dropdown
# -----------------------------
model_option = st.selectbox(
    "Select Model",
    (
        "KNN",
        "Decision Tree",
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    )
)

# -----------------------------
# Load selected model
# -----------------------------
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "KNN": "saved_models/knn_model.pkl",
        "Decision Tree": "saved_models/dt_model.pkl",
        "Logistic Regression": "saved_models/lr_model.pkl",
        "Random Forest": "saved_models/rf_model.pkl",
        "XGBoost": "saved_models/xgb_model.pkl"
    }
    return joblib.load(model_paths[model_name])

model = load_model(model_option)

st.success(f"{model_option} loaded successfully ✅")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV test dataset (semicolon separated)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file, sep=";")

    st.subheader("Uploaded Data Preview")
    st.dataframe(test_df.head())

    # Transform features
    X_transformed = pipeline.transform(test_df)

    # Predictions
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)[:, 1]

    results_df = test_df.copy()
    results_df["Prediction"] = predictions
    results_df["Probability (%)"] = (probabilities * 100).round(2)

    st.subheader("Prediction Results")
    st.dataframe(results_df.head())

    # Optional evaluation
    if "y" in test_df.columns:
        st.subheader("Evaluation Metrics")

        y_true = test_df["y"].map({"yes": 1, "no": 0})
        y_pred = predictions

        cm = confusion_matrix(y_true, y_pred)
        st.write("Confusion Matrix")
        st.write(cm)

        report = classification_report(y_true, y_pred)
        st.text("Classification Report")
        st.text(report)
