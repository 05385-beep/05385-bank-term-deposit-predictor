from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class FeaturePipelineBuilder:
    """
    Builds feature preprocessing pipeline and
    performs train-test splitting.
    """

    def __init__(self, test_ratio=0.25, random_seed=17):
        """
        Parameters
        ----------
        test_ratio : float
            Proportion of data to use for testing.
        random_seed : int
            Random seed for reproducibility.
        """
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.pipeline = None

    def split_features_target(self, df):
        """
        Separates input features and target variable.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        X : pd.DataFrame
            Feature columns
        y : pd.Series
            Target column
        """
        X = df.drop("y", axis=1)
        y = df["y"]
        return X, y

    def build_pipeline(self, X):
        """
        Builds preprocessing pipeline for numeric
        and categorical features.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        ColumnTransformer
            Preprocessing pipeline
        """
        categorical_cols = X.select_dtypes(include="object").columns
        numeric_cols = X.select_dtypes(exclude="object").columns

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
            ]
        )

        return self.pipeline

    def train_test_data(self, X, y):
        """
        Splits data into train and test sets.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X,
            y,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=y
        )
