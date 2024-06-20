from typing import List, Optional, Callable, Dict, Any, Tuple
import logging

import pandas as pd
import numpy as np
from numpy.typing import NDArray


class DatasetProperties:

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
        labels: Optional[List[str]] = None,
        benign_label: Optional[str] = None,
    ) -> None:
        self.features: List[str] = features
        self.categorical_features: List[str] = categorical_features
        self.numeric_features: List[str] = [
            feature for feature in features if feature not in categorical_features
        ]
        self.labels: Optional[List[str]] = labels
        self.benign_label: Optional[str] = benign_label

        logging.info(
            f"Dataset properties: Total features: {len(features)} (Numeric: {len(self.numeric_features)}, Categorical: {len(categorical_features)})\n"
        )


class Processor:

    __slots__ = [
        "_properties",
        "_df",
        "_transformations",
        "_transformations_mappings",
        "_train_mask",
        "_valid_mask",
        "_test_mask",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        properties: DatasetProperties,
        train_size: Optional[float] = 0.7,
        valid_size: Optional[float] = 0.1,
    ) -> None:
        self._properties: DatasetProperties = properties
        self._df: pd.DataFrame = df
        self._transformations: List[Callable] = []
        self._transformations_mappings: Dict[str, Dict[int, Any]] = {}

        train_mask, valid_mask, test_mask = self._df_split(train_size, valid_size)
        self._train_mask: pd.Series = pd.Series(train_mask, index=df.index)
        self._valid_mask: pd.Series = pd.Series(valid_mask, index=df.index)
        self._test_mask: pd.Series = pd.Series(test_mask, index=df.index)

    def _df_split(
        self, train_size: float, valid_size: float
    ) -> Tuple[NDArray, NDArray, NDArray]:
        total_samples = len(self._df)
        num_train = int(train_size * total_samples)
        num_valid = int(valid_size * total_samples)
        logging.info(
            f"Splitting {total_samples} samples into: train: {num_train}, valid: {num_valid}, test: {num_train - num_valid}..."
        )

        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        train_indices = indices[:num_train]
        valid_indices = indices[num_train : (num_train + num_valid)]
        test_indices = indices[(num_train + num_valid) :]

        train_mask = np.zeros(total_samples, dtype=bool)
        valid_mask = np.zeros(total_samples, dtype=bool)
        test_mask = np.zeros(total_samples, dtype=bool)

        train_mask[train_indices] = True
        valid_mask[valid_indices] = True
        test_mask[test_indices] = True

        return train_mask, valid_mask, test_mask

    @property
    def transformations(self) -> List[Callable]:
        return self._transformations

    @transformations.setter
    def transformations(self, transformations: List[Callable]) -> None:
        self._transformations = sorted(
            transformations, key=lambda transform_function: transform_function.order
        )

    def apply(self) -> None:
        logging.info(f"Applying {len(self._transformations)} transformation...")
        for transform_function in self._transformations:
            transform_map = transform_function(
                self._df, self._properties, self._train_mask
            )
            if transform_map is not None:
                self._transformations_mappings[transform_function.__name__] = (
                    transform_map
                )

        logging.info("Completed.\n")

    def get_train(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self._df[self._train_mask][self._properties.features],
            self._df[self._train_mask][self._properties.labels],
        )

    def get_valid(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self._df[self._valid_mask][self._properties.features],
            self._df[self._valid_mask][self._properties.labels],
        )

    def get_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return (
            self._df[self._test_mask][self._properties.features],
            self._df[self._test_mask][self._properties.labels],
        )

    def decode(self, data: int, map_name: str) -> Any:
        if map_name not in self._transformations_mappings:
            raise ValueError(f"Map {map_name} not in maps.")
        if data not in self._transformations_mappings[map_name]:
            raise ValueError(f"Value {data} not in {map_name}.")
        return self._transformations_mappings[map_name][data]

    def map_values(self, map_name: str) -> Any:
        if map_name not in self._transformations_mappings:
            raise ValueError(f"Map {map_name} not in maps.")
        return self._transformations_mappings[map_name].values()
