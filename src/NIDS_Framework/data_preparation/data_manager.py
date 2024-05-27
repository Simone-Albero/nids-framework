from typing import List, Optional, Callable, Any
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose

from data_preparation import custom_dataset


class DatasetProperties:
    """Provide the metadata of a dataset.

    Attributes:
        features: Fields to include as part of the training.
        categorical_features: Fields that should be treated as categorical.
        numeric_features: Fields that should be treated as numerical.
        labels: Fields treated as class labels.
        benign_label: The label for benign traffic.
    """

    __slots__ = [
        "features",
        "categorical_features",
        "numeric_features",
        "labels",
        "benign_label",
    ]

    def __init__(
        self,
        features: List[str],
        categorical_features: List[str],
        labels: Optional[str] = None,
        benign_label: Optional[str] = None,
    ) -> None:
        """Initialize the instance by specifying the unspecified numerical_features."""
        self.features: List[str] = features
        self.categorical_features: List[str] = categorical_features
        self.numeric_features: List[str] = [
            feature for feature in features if feature not in categorical_features
        ]
        self.labels: Optional[str] = labels
        self.benign_label: Optional[str] = benign_label


class DataManager:
    """Manage the original dataset, generating train-set and test-set.

    Attributes:
        _train_df: The training set as Pandas DataFrame.
        _test_df: The test set as Pandas DataFrame.
        _properties: The dataset metadata.
        _numeric_transformations: The list of numerical data transformations.
        _categorical_transformations: The list of categorical data transformations.
        _target_transformation: The list of labels transformations.
    """

    __slots__ = [
        "_train_df",
        "_test_df",
        "_properties",
        "_numeric_transformations",
        "_categorical_transformations",
        "_target_transformation",
    ]

    def __init__(
        self,
        dataset_path: str = None,
        properties: DatasetProperties = None,
    ) -> None:
        self._properties: DatasetProperties = properties

        df = self._load_df(dataset_path)
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        self._train_df: pd.DataFrame = train_df
        self._test_df: pd.DataFrame = test_df
        self._numeric_transformations = []
        self._categorical_transformations = []
        self._target_transformation = []

    def _load_df(self, dataset_path: str) -> pd.DataFrame:
        """Load the dataset from file.

        Args:
            dataset_path: The path to the file.

        Raises:
            FileNotFoundError: Raised if the dataset path does not exist.
            NotImplementedError: Raised if the dataset format differs from CSV.

        Returns:
            The dataset as a pandas DataFrame.
        """

        if not dataset_path.lower().endswith(".csv"):
            raise NotImplementedError("Only CSV files are supported.")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"The dataset file was not found at the specified path: {dataset_path}"
            )

        logging.info(f"Reading dataset from: {dataset_path}.")
        return pd.read_csv(dataset_path)

    def train_data(self) -> custom_dataset.CustomDataset:
        """Provide the torch Dataset generated from train data."""
        dataset = custom_dataset.CustomDataset(
            self._train_df[self._properties.numeric_features],
            self._train_df[self._properties.categorical_features],
            self._train_df[self._properties.labels],
            Compose(sorted(self._numeric_transformations, key=lambda x: x.priority)),
            Compose(sorted(self._categorical_transformations, key=lambda x: x.priority)),
            Compose(sorted(self._target_transformation, key=lambda x: x.priority)),
        )
        return dataset

    def test_data(self) -> custom_dataset.CustomDataset:
        """Provide the torch Dataset generated from test data."""
        return custom_dataset.CustomDataset(self._test_df)
    
    def numeric_transformation(self, priority: int):
        def decorator(transform_function):
            def transform_sample(sample):
                return transform_function(sample)

            transform_sample.priority = priority
            self._numeric_transformations.append(transform_sample)
            return transform_sample

        return decorator
    
    def categorical_transformation(self, priority: int):
        def decorator(transform_function):
            def transform_sample(sample):
                return transform_function(sample)

            transform_sample.priority = priority
            self._categorical_transformations.append(transform_sample)
            return transform_sample

        return decorator
    
    def target_transformation(self, priority: int):
        def decorator(transform_function):
            def transform_sample(sample):
                return transform_function(sample)

            transform_sample.priority = priority
            self._target_transformation.append(transform_sample)
            return transform_sample

        return decorator
