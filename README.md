# 🏦 Bank Marketing Term Deposit Prediction

## Project Links

- **GitHub Repository:**  
  https://github.com/05385-beep/05385-bank-term-deposit-predictor  

- **Live Streamlit App:**  
  https://05385-bank-term-deposit-predictor-7wn9t7scljzxzslh5gfzab.streamlit.app/

---

#  Problem Statement

The objective of this project is to predict whether a customer will subscribe to a term deposit based on historical bank marketing campaign data.

This is a **binary classification problem**, where:

- `Yes` → Customer subscribed to term deposit  
- `No` → Customer did not subscribe  

Multiple classification algorithms were implemented and compared using various evaluation metrics to determine the best-performing model.

---

# Dataset Description

- **Dataset:** Bank Marketing Dataset (UCI Machine Learning Repository)
- **Total Instances:** 45,000+
- **Total Features:** 16
- **Target Variable:** `y` (yes/no)

### Feature Types
- Numerical Features: age, duration, campaign, previous, etc.
- Categorical Features: job, marital, education, contact, month, etc.

### Preprocessing Steps
- One-hot encoding for categorical variables
- Standard scaling for numerical features
- Train-test split
- Model persistence using joblib

---

#  Models Implemented

The following six classification models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

All models were evaluated using identical preprocessing pipelines.

---

# Evaluation Metrics Used

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

#  Model Comparison Table

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.9125 | 0.9378 | 0.6724 | 0.4353 | 0.5285 | 0.4966 |
| Decision Tree | 0.9133 | 0.9300 | 0.6256 | 0.5733 | 0.5983 | 0.5504 |
| KNN | 0.9023 | 0.8724 | 0.5883 | 0.4422 | 0.5049 | 0.4576 |
| Naive Bayes | 0.8392 | 0.8496 | 0.3714 | 0.6172 | 0.4637 | 0.3928 |
| Random Forest | 0.9147 | 0.9441 | 0.6539 | 0.5164 | 0.5771 | 0.5350 |
| XGBoost | 0.9162 | 0.9512 | 0.6511 | 0.5517 | 0.5973 | 0.55


# Observations

- **XGBoost** achieved the highest AUC (0.9512) and MCC (0.5532), making it the best overall performing model.
- **Random Forest** also performed strongly due to ensemble learning and reduced overfitting.
- **Decision Tree** showed improved recall compared to Logistic Regression.
- **Logistic Regression** achieved strong AUC but relatively lower recall.
- **Naive Bayes** demonstrated higher recall but lower precision due to independence assumptions.
- **KNN** performed moderately compared to tree-based ensemble models.

Overall, ensemble models outperformed individual classifiers on this dataset.

---

# Streamlit Application Features

The deployed Streamlit application includes:

- 📂 CSV Dataset Upload
- 🔄 Model Selection
- 🎚 Classification Threshold Slider
- 📊 Display of Evaluation Metrics
- 🔲 Confusion Matrix Visualization
- 🎨 Highlighted Prediction Table
- 📥 Sample Test Data Download Links

The app supports both:
- Test datasets WITH target column (for metrics display)
- Test datasets WITHOUT target column (for pure prediction)

---
