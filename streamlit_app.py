import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import os

st.set_page_config(page_title="Bank Marketing - KNN", layout="wide")

st.title("📊 Bank Marketing Term Deposit Prediction (KNN)")
st.write(
    "This application uses a trained **K-Nearest Neighbors (KNN)** model "
    "to predict whether a customer will subscribe to a term deposit."
)

# -----------------------------
# Load model and preprocessing
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("saved_models/knn_model.pkl")
    pipeline = joblib.load("saved_models/knn_pipeline.pkl")
    return model, pipeline

model, pipeline = load_artifacts()

# -----------------------------
# File upload
# -----------------------------
st.header("Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV file (semicolon separated, same columns as training data)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file, sep=";")

    st.subheader("Preview of Uploaded Data")
    st.dataframe(test_df.head())

    # -----------------------------
    # Prediction
    # -----------------------------
    X_transformed = pipeline.transform(test_df)
    predictions = model.predict(X_transformed)
    probabilities = model.predict_proba(X_transformed)[:, 1]

    results_df = test_df.copy()
    results_df["Prediction"] = predictions
    results_df["Probability"] = probabilities

    st.subheader("Prediction Results (Sample)")
    st.dataframe(results_df.head())

    # -----------------------------
    # Evaluation (optional)
    # -----------------------------
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
