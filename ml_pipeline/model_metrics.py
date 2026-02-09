from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)


class BinaryClassificationMetrics:
    """
    Computes evaluation metrics for
    binary classification models.
    """

    @staticmethod
    def evaluate(y_true, y_pred, y_prob):
        """
        Calculates all required metrics.

        Parameters
        ----------
        y_true : array-like
            True class labels
        y_pred : array-like
            Predicted class labels
        y_prob : array-like
            Predicted probabilities for positive class

        Returns
        -------
        dict
            Dictionary containing evaluation metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_prob),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }
        return metrics
