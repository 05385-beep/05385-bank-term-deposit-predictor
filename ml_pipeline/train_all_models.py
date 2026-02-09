import os
import joblib

from ml_pipeline.data_loader import BankMarketingLoader
from ml_pipeline.feature_builder import FeaturePipelineBuilder
from ml_pipeline.model_trainer import KNNModel
from ml_pipeline.model_metrics import BinaryClassificationMetrics


def run_knn_pipeline():
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

    # 5. Train KNN model
    knn = KNNModel(neighbors=5)
    knn.build()
    knn.train(X_train_transformed, y_train)

    # 6. Predictions
    y_pred = knn.predict(X_test_transformed)
    y_prob = knn.predict_proba(X_test_transformed)

    # 7. Evaluation
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
    joblib.dump(pipeline, "saved_models/knn_pipeline.pkl")


if __name__ == "__main__":
    run_knn_pipeline()
