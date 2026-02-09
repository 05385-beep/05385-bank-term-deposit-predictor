from sklearn.neighbors import KNeighborsClassifier
import joblib


class KNNModel:
    """
    K-Nearest Neighbors classifier wrapper.
    """

    def __init__(self, neighbors=5):
        self.neighbors = neighbors
        self.model = None

    def build(self):
        self.model = KNeighborsClassifier(
            n_neighbors=self.neighbors,
            weights="distance"
        )
        return self.model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def save(self, path):
        joblib.dump(self.model, path)
