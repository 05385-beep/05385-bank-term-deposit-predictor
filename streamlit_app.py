import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="Bank Marketing ML Models", layout="wide")

st.title("📊 Bank Marketing Term Deposit Prediction")
st.write(
    "Upload a test dataset and select a model to generate predictions. "
    "Adjust the classification threshold to observe precision-recall tradeoff."
)

# ---------------------------------
# Load Preprocessing Pipeline
# ---------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("saved_models/preprocessing_pipeline.pkl")

pipeline = load_pipeline()

# ---------------------------------
# Model Selection Dropdown
# ---------------------------------
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

# ---------------------------------
# Load Selected Model
# ---------------------------------
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

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV test dataset (semicolon separated)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file, sep=";")

    st.subheader("Uploaded Data Preview")

    # Row Display Slider
    row_count = st.slider(
        "Select number of rows to display",
        min_value=5,
        max_value=min(len(test_df), 100),
        value=min(10, len(test_df))
    )

    st.dataframe(test_df.head(row_count))

    # ---------------------------------
    # Feature Transformation
    # ---------------------------------
    X_transformed = pipeline.transform(test_df)

    # ---------------------------------
    # Probability Predictions
    # ---------------------------------
    probabilities = model.predict_proba(X_transformed)[:, 1]

    # Threshold Slider
    threshold = st.slider(
        "Select Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )

    # Apply Threshold
    predictions = (probabilities >= threshold).astype(int)

    # ---------------------------------
    # Results DataFrame
    # ---------------------------------
    results_df = test_df.copy()
    results_df["Probability (%)"] = (probabilities * 100).round(2)
    results_df["Prediction"] = predictions
    results_df["Prediction"] = results_df["Prediction"].map({1: "Yes", 0: "No"})

    st.subheader("Prediction Results")
    st.dataframe(results_df.head(row_count))

    # ---------------------------------
    # Evaluation Metrics (if y exists)
    # ---------------------------------
    if "y" in test_df.columns:
        st.subheader("📈 Evaluation Metrics")

        y_true = test_df["y"].map({"yes": 1, "no": 0})
        y_pred = predictions

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, probabilities)
        mcc = matthews_corrcoef(y_true, y_pred)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("AUC", f"{auc:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

        st.subheader("Classification Report")
        report = classification_report(y_true, y_pred)
        st.text(report)
