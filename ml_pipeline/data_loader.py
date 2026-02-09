import pandas as pd


class BankMarketingLoader:
    """
    Handles loading and basic preparation of the
    Bank Marketing dataset.
    """

    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path : str
            Path to the Bank Marketing CSV file.
        """
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from CSV.

        Returns
        -------
        pd.DataFrame
            Raw dataset as a pandas DataFrame.
        """
        df = pd.read_csv(self.file_path, sep=';')
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes the target column 'y' into numeric form.

        yes -> 1
        no  -> 0

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with target column.

        Returns
        -------
        pd.DataFrame
            Dataframe with encoded target.
        """
        df = df.copy()
        df["y"] = df["y"].map({"yes": 1, "no": 0})
        return df
