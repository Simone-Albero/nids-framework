from typing import List, Optional, Any
import os
import math

import pandas as pd
import numpy as np

import utilities


class DatasetProperties(object):
    """Define the format of a specific dataset.

    Attributes:
        features: Fields to include as part of the training.
        categorical_features: Fields that should be treated as categorical.
        numerical_features: Fields that should be treated as numerical.
        labels_column: Fields treated as class labels.
        benign_label: The label for benign traffic.
    """

    __slots__ = [
        "features",
        "categorical_features",
        "numerical_features",
        "labels_column",
        "benign_label",
    ]

    def __init__(
        self,
        features: List[str],
        categorical_features: List[str],
        labels_column: Optional[str] = None,
        benign_label: Optional[str] = None,
    ) -> None:
        """Initialize the instance by specifying the unspecified numerical_features."""
        self.features: List[str] = features
        self.categorical_features: List[str] = categorical_features
        self.numerical_features: List[str] = [
            feature for feature in features if feature not in categorical_features
        ]
        self.labels_column: Optional[str] = labels_column
        self.benign_label: Optional[str] = benign_label


class Dataset(object):
    """Represent the abstraction of a dataset.

    Attributes:
        cache_path: The cache file path.
        _df: The dataset content as a Pandas DataFrame.
        _properties: The dataset format.
        _training_mask: The indices for training smaples.
    """

    __slots__ = ["cache_path", "_df", "_properties", "_training_mask"]

    def __init__(
        self,
        cache_path: str,
        dataset_path: Optional[str] = None,
        properties: Optional[DatasetProperties] = None,
    ) -> None:
        """Initialize the instance from the cache file (if it exists), otherwise using the dataset path and format passed as arguments.

        Raises:
            ValueError: Raised if neither the cache path nor the file path is specified.
        """
        self.cache_path: str = cache_path
        self._df: Optional[pd.DataFrame] = None
        self._properties: Optional[DatasetProperties] = None
        self._training_mask: np.ndarray[int] = None

        if os.path.exists(cache_path):
            self._load_from_cache()

        elif dataset_path is not None:
            self._load_from_file(dataset_path, properties)

        else:
            raise ValueError(
                f"Please provide either the cache path or the dataset path!"
            )

    class ColumnIterator(object):
        """Iterator iterating over the columns of the DataFrame.

        Attributes:
            _df: The DataFrame to iterate over.
            _features: The column names to iterate over.
            _sample_mask: The index of rows to consider within the context of the column.
            _index: The current iterator index.
        """

        __slots__ = ["_df", "_features", "_sample_mask", "_index"]

        def __init__(
            self,
            df: pd.DataFrame,
            features: List[str],
            sample_mask: Optional[np.ndarray[int]] = None,
        ) -> None:
            self._df = df
            self._features = features
            self._sample_mask = sample_mask
            self._index = 0

        def __iter__(self):
            """Return iterator object."""
            return self

        def __next__(self):
            """Return the next column name and its corresponding values."""
            while self._index < len(self._features):
                column_name = self._features[self._index]
                self._index += 1

                if self._sample_mask is None:
                    return (column_name, self._df[column_name].values)
                else:
                    return (column_name, self._df[column_name][self._sample_mask].values)
            raise StopIteration

    def _load_from_file(self, dataset_path: str, properties: DatasetProperties) -> None:
        """Load dataset from file and save to cache.

        Args:
            dataset_path: The path to the dataset file.
            properties: The format of the dataset.

        Raises:
            FileNotFoundError: Raised if the dataset path does not exist.
            NotImplementedError: Raised if the dataset format differs from CSV.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        print(f"Reading dataset from path '{dataset_path}' ...")
        if not dataset_path.lower().endswith(".csv"):
            raise NotImplementedError("Only CSV files are supported.")

        self._df = pd.read_csv(dataset_path)
        self._properties = properties
        self._save_to_cache()

    def _save_to_cache(self) -> None:
        """Save the Dataset to a file used as cache."""
        data = {"df": self._df, "properties": self._properties}
        utilities.file_write(self.cache_path, data)

    def _load_from_cache(self) -> None:
        """Load the Dataset from a file used as cache."""
        data = utilities.file_read(self.cache_path)
        self._df, self._properties = data["df"], data["properties"]

    def column_iterator(self) -> ColumnIterator:
        """Create an iterator to iterate over all columns of the dataset.

        Returns:
            The iterator to iterate over.
        """
        return self.ColumnIterator(self._df, self._properties.features)

    def numerical_column_iterator(self, isOverTrain: bool = False) -> ColumnIterator:
        """Create an iterator to iterate over the numerical columns of the dataset.

        Args:
            isOverTrain: If True, limit the number of columns considered by the iterator to those specified by '_training_mask'.

        Returns:
            The iterator to iterate over.
        """
        if isOverTrain:
            return self.ColumnIterator(
                self._df, self._properties.numerical_features, self._training_mask
            )
        else:
            return self.ColumnIterator(
                self._df, self._properties.numerical_features
            )

    def categorical_column_iterator(self, isOverTrain: bool = False) -> ColumnIterator:
        """Create an iterator to iterate over the categorical columns of the dataset.

        Args:
            isOverTrain: If True, limit the number of columns considered by the iterator to those specified by '_training_mask'.

        Returns:
            The iterator to iterate over.
        """
        if isOverTrain:
            return self.ColumnIterator(
                self._df, self._properties.categorical_features, self._training_mask
            )
        else:
            return self.ColumnIterator(
                self._df, self._properties.categorical_features
            )

    def update_column(self, column_name: str, values: List[Any]) -> None:
        """Update the values of the specified Dataset column.

        Args:
            column_name: The name of the column to update.
            values: The new values to assign to the column.
        
        Raises:
            ValueError: Raised if values length differs from Dataset column length.
        """
        if len(values) != self._df.shape[0]:
            raise ValueError(f"The size of the values, {len(values)}, does not match the size of the dataset columns, {self._df.shape[0]}.")

        self._df[column_name] = values

    def drop_column(self, column_name: str) -> None:
        """Remove the column named column_name from the Dataset.

        Args:
            column_name: The name of the column to remove.
        
        Raises:
            ValueError: Raised if 'column_name' does not exsit between the Dataset columns.
        """
        if column_name not in self._df.columns:
            raise ValueError(f"Column: {column_name} does not exist.")

        self._df.drop(columns=[column_name], inplace=True)
    
    def random_train_test_split(self, train_test_ratio: float) -> None:
        """Randomly generate the training mask used to distinguish training set samples from test set samples.

        Args:
            train_test_ratio: The ratio between the size of the training set and the test set.
        """
        df_size = self._df.shape[0]
        self._training_mask = np.zeros(df_size, dtype=int)
        train_size = math.floor(train_test_ratio * df_size)

        random_samples = np.random.choice(df_size, train_size, replace=False)
        self._training_mask[random_samples] = 1