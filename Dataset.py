import pandas as pd
from typing import Self
import numpy as np
import logging


class Dataset:

    def __init__(self, data: pd.DataFrame) -> None:
        logging.info(f"Creating Dataset object from {type(data)}")
        self.data: pd.DataFrame = data.copy()
        self.split: bool = False
        self.cleaned: bool = False

        # para lidiar con datos nuevos a evaluar, se han de eliminar las columnas que no se usaron en el entrenamiento y usar las medias y stds de los datos de entrenamiento
        self.cols_to_drop: list[str] = []
        self.train_means: dict[str, float] = {}
        self.train_stds: dict[str, float] = {}

    def __str__(self) -> str:
        l1: str = f"Split dataframe: {self.split}\n"
        l2: str = f"Clean dataframe: {self.cleaned}\n"
        return l1 + l2

    def auto(self,target_name)->Self:
        """
        This method is used to automatically clean the data. It will remove columns with more than 10% missing values, standarize columns and split the data into train and test sets.
        """
        self.rename_target(target_name=target_name)
        self.one_hot_encode()
        self.split_train_test()
        self.clean_missing(missing_threshold=0.1)
        self.standarize()
        self.split_x_y()
        return self

    def clean_missing(
        self, missing_threshold: float = 0.1, test_method: str = "delete"
    ) -> Self:

        if self.split == False:
            raise ValueError(
                "Data has not been split. Split in train test before cleaning missing data."
            )

        logging.info(
            f"Cleaning missing data with threshold {missing_threshold} and method {test_method}"
        )
        cols_to_be_removed = []  # cols to be removed given the train data

        for column in self.train.columns:
            if (
                self.train[column].isna().sum() / self.train.shape[0]
                > missing_threshold
            ):
                self.train.drop(column, axis=1, inplace=True)
                cols_to_be_removed.append(column)

        self.train.dropna(inplace=True)
        self.cols_to_drop += cols_to_be_removed

        for column in cols_to_be_removed:
            self.test.drop(column, axis=1, inplace=True)

        if test_method == "delete":
            self.test.dropna(inplace=True)

        elif test_method == "mean":
            for column in self.test.columns:

                mean_value: float = self.train[column].mean()
                self.test.fillna({column: mean_value}, inplace=True)

        elif test_method == "median":
            for column in self.test.columns:
                median_value: float = self.train[column].median()
                self.test.fillna({column: median_value}, inplace=True)

        self.cleaned = True
        logging.info(f"Columns droped: {cols_to_be_removed}")

        return self

    def create_cuadratic_features(self) -> Self:
        if self.cleaned == False:
            raise ValueError(
                "Data has no been split. Clean data before to reduce the number of features."
            )

        columns: list[str] = self.train.select_dtypes(
            exclude=["object"]
        ).columns.tolist()
        if "target" in columns:
            columns.remove("target")

        logging.info(f"Creating cuadratic features from columns {columns}")

        n = len(columns)

        for i in range(n):
            for j in range(i, n):
                if i == j and self.train[columns[i]].dtype == "bool":
                    continue
                self.train[f"{i}_{j}"] = self.train[columns[i]] * self.train[columns[j]]
                self.test[f"{i}_{j}"] = self.test[columns[i]] * self.test[columns[j]]
        return self

    def create_rate_features(
        self, time_feature: str, cols_to_ignore: list[str] = ["target"]
    ) -> Self:
        if self.cleaned == False:
            raise ValueError(
                "Data has no been split. Clean data before to reduce the number of features."
            )

        columns: list[str] = df.select_dtypes(include=["number"]).columns.tolist()

        for column in columns:
            if column != time_feature and column not in cols_to_ignore:
                self.train[f"{column}_rate"] = (
                    self.train[column] / self.train[time_feature]
                )
                self.test[f"{column}_rate"] = (
                    self.test[column] / self.test[time_feature]
                )

        return self

    def get_health(self, verbose: bool = False) -> dict[str, int]:

        missing: dict = {}
        for column in self.data.columns:
            missing[column] = int(self.data[column].isna().sum())
            if verbose:
                print(f"{column}\t {missing[column]}")
        missing_rate = sum(missing.values()) / (self.data.shape[0] * self.data.shape[1])
        self.missing = missing
        self.missing_rate = missing_rate

        return missing

    def drop_columns(self, cols_to_drop: list[str]) -> Self:
        self.cols_to_drop += cols_to_drop
        if self.split:
            raise ValueError("Data has already been split. Cannot drop columns.")

        logging.info(f"Dropping columns {cols_to_drop} from dataset")

        for column in self.data.columns:
            if column in cols_to_drop:
                self.data.drop(column, axis=1, inplace=True)

        return self

    def normalize(
        self, max_value: float = 1, cols_to_ignore: list[str] = ["target"]
    ) -> Self:
        if self.split == False:
            raise ValueError("Data has not been split. Cannot normalize dataset.")

        logging.info(f"Normalizing data with max value {max_value}")
        for column in self.train.columns:
            if column in cols_to_ignore:
                continue
            if (
                self.data[column].dtype == "float64"
                or self.train[column].dtype == "int64"
            ):
                column_max = self.train[column].max()
                column_min = self.train[column].min()

                self.train[column] = (
                    (self.train[column] - column_min)
                    / (column_max - column_min)
                    * max_value
                )
                self.test[column] = (
                    (self.test[column] - column_min)
                    / (column_max - column_min)
                    * max_value
                )

        return self

    def one_hot_encode(self) -> Self:
        if self.split:
            raise ValueError("Data has already been split. Cannot one hot encode.")

        for column in self.data.columns:
            if self.data[column].dtype == "object":
                logging.info(f"One hot encoding column {column}")
                self.data = pd.get_dummies(
                    self.data, columns=[column], drop_first=True, dtype=bool
                )

        return self

    def rename_target(self, target_name: str) -> Self:
        if self.split:
            raise ValueError("Data has already been split. Cannot rename target.")
        if target_name not in self.data.columns:
            raise ValueError(f"Column {target_name} not found in the DataFrame.")

        logging.info(f"Renaming target column {target_name} to 'target'")
        self.data.rename(columns={target_name: "target"}, inplace=True)

        return self

    def remove_outliers(
        self, threshold_n_sigmas: float, cols_to_ignore: list[str] = []
    ) -> Self:

        if self.split == False:
            raise ValueError(
                "Data has not been split in train/test. Cannot remove outliers."
            )

        for column in self.train.columns:
            if column in cols_to_ignore:
                continue
            if self.train[column].dtype == "float64":
                mean = self.train[column].mean()
                std = self.train[column].std()
                self.train = self.train[
                    np.abs(self.train[column] - mean) <= threshold_n_sigmas * std
                ]
                self.test = self.test[
                    np.abs(self.test[column] - mean) <= threshold_n_sigmas * std
                ]

        return self

    def split_train_test(
        self, seed_number: int = 0, train_test_split: float = 0.8
    ) -> Self:
        if self.split == True:
            raise ValueError("Data has already been split")
        np.random.seed(seed_number)
        train_mask: np.ndarray = np.random.rand(len(self.data)) < train_test_split
        self.train: pd.DataFrame = self.data[train_mask].copy()
        self.test: pd.DataFrame = self.data[~train_mask].copy()
        self.split = True
        self.data = None

        return self

    def split_x_y(self) -> Self:
        if self.split == False:
            raise ValueError(
                "Data has not been split in train and test. Cannot split into x and y."
            )
        logging.info("Splitting data into x and y")
        self.train_x = self.train.drop(columns=["target"])
        self.train_y = self.train["target"]
        self.test_x = self.test.drop(columns=["target"])
        self.test_y = self.test["target"]
        return self

    def standarize(self, cols_to_ignore: list[str] = ["target"]) -> Self:
        if self.split == False:
            raise ValueError("Data has not been split. Cannot standarize.")

        logging.info(f"Standarizing data with cols to ignore {cols_to_ignore}")

        for column in self.train.columns:
            if column in cols_to_ignore:
                continue
            if (
                self.train[column].dtype == "float64"
                or self.train[column].dtype == "int64"
            ):
                # Explicitly cast the column to float
                self.train[column] = self.train[column].astype("float64")
                self.test[column] = self.test[column].astype("float64")

                train_mean: float = float(self.train[column].mean())
                train_std: float = float(self.train[column].std())
                self.train_means[column] = train_mean
                self.train_stds[column] = train_std

                self.train[column] = (self.train[column] - train_mean) / train_std
                self.test[column] = (self.test[column] - train_mean) / train_std
        return self


if __name__ == "__main__":

    df = pd.DataFrame()
    df["c1"] = [0, 1, 2, 3, 4, 1, 2, 3, 4, 2, 1, 2, 3, 2, 3]
    df["c2"] = [5, 6, 7, 8, 9, 2, 3, 4, 7, 6, 4, 3, 5, 4, 5]
    df["c3"] = [1, 2, 3, 3, 4, 3, 4, 5, 6, 3, 5, None, 5, 4, 3]
    df["c4"] = [1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, None, 5, None, 12]
    df["c5"] = [
        "a",
        "a",
        "b",
        "b",
        "c",
        "a",
        "a",
        "b",
        "b",
        "c",
        "a",
        "a",
        "b",
        "b",
        "c",
    ]
    dataset = Dataset(data=df)

    dataset.auto(target_name="c1")

    print(dataset.train)
    print(dataset.test)
