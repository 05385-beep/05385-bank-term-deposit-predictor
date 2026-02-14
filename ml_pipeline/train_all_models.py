import os
import joblib

from ml_pipeline.data_loader import BankMarketingLoader
from ml_pipeline.feature_builder import FeaturePipelineBuilder
from ml_pipeline.model_trainer import KNNModel, DecisionTreeModel, LogisticRegressionModel, NaiveBayesModel, RandomForestModel, XGBoostModel
from ml_pipeline.model_metrics import BinaryClassificationMetrics


def train_all_models():
    """
    Executes end-to-end KNN training and evaluation.
    """

    # 1. Load and prepare dataset
    loader = BankMarketingLoader("data/bank_marketing_full.csv")
    df = loader.load_data()
    df = loader.encode_target(df)

    # 2. Feature engineering
    builder = FeaturePipelineBuilder()
    X, y = builder.split_features_target(df)
    pipeline = builder.build_pipeline(X)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = builder.train_test_data(X, y)

    # 4. Apply preprocessing
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    #Train KNN model
    knn = KNNModel(neighbors=5)
    knn.build()
    knn.train(X_train_transformed, y_train)

    # Predictions
    y_pred = knn.predict(X_test_transformed)
    y_prob = knn.predict_proba(X_test_transformed)

    # Evaluation
    metrics = BinaryClassificationMetrics.evaluate(
        y_test, y_pred, y_prob
    )

    print("KNN Evaluation Metrics")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 8. Save model and preprocessing pipeline
    os.makedirs("saved_models", exist_ok=True)
    knn.save("saved_models/knn_model.pkl")


    # ------------------------------------------------
    # Decision Tree
    # ------------------------------------------------
    dt = DecisionTreeModel(max_depth=6)
    dt.build()
    dt.train(X_train_transformed, y_train)

    y_pred_dt = dt.predict(X_test_transformed)
    y_prob_dt = dt.predict_proba(X_test_transformed)

    metrics_dt = BinaryClassificationMetrics.evaluate(
        y_test, y_pred_dt, y_prob_dt
    )

    print("\nDecision Tree Metrics")
    print("-" * 30)
    for k, v in metrics_dt.items():
        print(f"{k}: {v:.4f}")

    dt.save("saved_models/dt_model.pkl")

    # ==============================
    # Logistic Regression
    # ==============================
    lr = LogisticRegressionModel()
    lr.build()
    lr.train(X_train_transformed, y_train)

    y_pred_lr = lr.predict(X_test_transformed)
    y_prob_lr = lr.predict_proba(X_test_transformed)

    metrics_lr = BinaryClassificationMetrics.evaluate(
        y_test, y_pred_lr, y_prob_lr
    )

    print("\nLogistic Regression Metrics")
    print("-" * 30)
    for k, v in metrics_lr.items():
        print(f"{k}: {v:.4f}")

    lr.save("saved_models/lr_model.pkl")
    
    # ==============================
    # Naive Bayes
    # ==============================
    nb = NaiveBayesModel()
    nb.build()
    nb.train(X_train_transformed, y_train)

    y_pred_nb = nb.predict(X_test_transformed)
    y_prob_nb = nb.predict_proba(X_test_transformed)

    metrics_nb = BinaryClassificationMetrics.evaluate(
        y_test, y_pred_nb, y_prob_nb
    )

    print("\nNaive Bayes Metrics")
    print("-" * 30)
    for k, v in metrics_nb.items():
        print(f"{k}: {v:.4f}")

    nb.save("saved_models/nb_model.pkl")
    

    # ==============================
    # Random Forest
    # ==============================
    rf = RandomForestModel(n_estimators=200)
    rf.build()
    rf.train(X_train_transformed, y_train)
    y_pred_rf = rf.predict(X_test_transformed)
    y_prob_rf = rf.predict_proba(X_test_transformed)
    metrics_rf = BinaryClassificationMetrics.evaluate(y_test, y_pred_rf, y_prob_rf)

    print("\nRandom Forest Metrics")
    print("-" * 30)
    for k, v in metrics_rf.items():
        print(f"{k}: {v:.4f}")
    rf.save("saved_models/rf_model.pkl")

    # ==============================
    # XGBoost
    # ==============================
    xgb = XGBoostModel()
    xgb.build()
    xgb.train(X_train_transformed, y_train)

    y_pred_xgb = xgb.predict(X_test_transformed)
    y_prob_xgb = xgb.predict_proba(X_test_transformed)

    metrics_xgb = BinaryClassificationMetrics.evaluate(
        y_test, y_pred_xgb, y_prob_xgb
    )

    print("\nXGBoost Metrics")
    print("-" * 30)
    for k, v in metrics_xgb.items():
        print(f"{k}: {v:.4f}")

    xgb.save("saved_models/xgb_model.pkl")


    joblib.dump(pipeline, "saved_models/preprocessing_pipeline.pkl")

if __name__ == "__main__":
    train_all_models()

