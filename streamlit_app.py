import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
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
st.set_page_config(page_title="Bank Marketing ML Dashboard", layout="wide")

st.title("🏦 Bank Marketing Term Deposit Prediction Dashboard")
st.write(
    "Select a model, adjust classification threshold, and upload a test dataset "
    "to generate predictions."
)

# ---------------------------------
# Sidebar Controls
# ---------------------------------
st.sidebar.header("⚙️ Model Settings")

model_option = st.sidebar.radio(
    "Select Model",
    ["KNN", "Decision Tree", "Logistic Regression", "Naive Bayes", "Random Forest", "XGBoost"]
)

threshold = st.sidebar.slider(
    "Classification Threshold",
    0.1, 0.9, 0.5, 0.05
)

row_count = st.sidebar.slider(
    "Rows to Display",
    5, 1000, 500
)


# ---------------------------------
# Load Resources
# ---------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("model/preprocessing_pipeline.pkl")

@st.cache_resource
def load_model(model_name):
    model_paths = {
        "KNN": "model/knn_model.pkl",
        "Decision Tree": "model/dt_model.pkl",
        "Logistic Regression": "model/lr_model.pkl",
        "Naive Bayes": "model/nb_model.pkl",
        "Random Forest": "model/rf_model.pkl",
        "XGBoost": "model/xgb_model.pkl"
    }
    return joblib.load(model_paths[model_name])

pipeline = load_pipeline()
model = load_model(model_option)

st.sidebar.success(f"{model_option} Loaded Successfully")

# ---------------------------------
# GitHub Download Links
# ---------------------------------
st.markdown("### 📥 Download Sample Test Files")
st.markdown(
    """
- 🔹 [Download Test Data WITH 'y' (For Metrics)](https://raw.githubusercontent.com/05385-beep/05385-bank-term-deposit-predictor/main/data/bank_test_data_with_y.csv)

- 🔹 [Download Test Data WITHOUT 'y' (For Prediction)](https://raw.githubusercontent.com/05385-beep/05385-bank-term-deposit-predictor/main/data/bank_test_data_without_y.csv)
"""
)

st.markdown("---")

# ---------------------------------
# File Upload
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV Test Dataset (semicolon separated)",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file, sep=";")

    # ---------------------------------
    # Feature Transformation
    # ---------------------------------
    X_transformed = pipeline.transform(test_df)

    # ---------------------------------
    # Probability Predictions
    # ---------------------------------
    probabilities = model.predict_proba(X_transformed)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # ---------------------------------
    # Prepare Results
    # ---------------------------------
    results_df = test_df.copy()
    results_df["Probability (%)"] = (probabilities * 100).round(2)
    results_df["Prediction"] = predictions
    results_df["Prediction"] = results_df["Prediction"].map({1: "Yes", 0: "No"})

    # ---------------------------------
    # Tabs Layout
    # ---------------------------------
    tab1, tab2, tab3 = st.tabs(["📄 Predictions", "📊 Metrics", "🔲 Confusion Matrix"])

   # -----------------------------
# Tab 1: Predictions
# -----------------------------
with tab1:
    st.subheader("Prediction Results")

    # Inform user about sorting
    st.info("🔎 Table is default sorted to show 'Yes' predictions on top.")

    # Move Prediction and Probability to front
    cols = ["Prediction", "Probability (%)"] + \
           [col for col in results_df.columns if col not in ["Prediction", "Probability (%)"]]

    display_df = results_df[cols]

    # Default sort: Yes first
    display_df = display_df.sort_values(
        by="Prediction",
        ascending=False   # Yes comes before No (since Yes > No alphabetically)
    )

    display_df = display_df.head(row_count)

    # Styling function
    def highlight_prediction(val):
        if val == "Yes":
            return "background-color: #d4edda; color: black;"
        else:
            return "background-color: #f8d7da; color: black;"

    styled_df = display_df.style \
        .applymap(highlight_prediction, subset=["Prediction"]) \
        .background_gradient(
            subset=["Probability (%)"],
            cmap="YlGn"
        )

    st.dataframe(styled_df, use_container_width=True)



    # -----------------------------
    # Tab 2: Metrics
    # -----------------------------
    with tab2:
        if "y" in test_df.columns:
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
        else:
            st.info("Upload dataset WITH 'y' column to view evaluation metrics.")

    # -----------------------------
    # Tab 3: Confusion Matrix
    # -----------------------------
    with tab3:
        if "y" in test_df.columns:
            y_true = test_df["y"].map({"yes": 1, "no": 0})
            cm = confusion_matrix(y_true, predictions)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig)
        else:
            st.info("Upload dataset WITH 'y' column to view confusion matrix.")
