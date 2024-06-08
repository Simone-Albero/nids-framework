from typing import List, Optional, Callable, Dict, Any, Tuple
import os
import logging
import functools

import pandas as pd
import numpy as np


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
            f"Dataset properties: Total features: {len(features)} (Numeric: {len(self.numeric_features)}, Categorical: {len(categorical_features)})"
        )


class Processor:

    __slots__ = [
        "_properties",
        "_df",
        "_transformations",
        "_label_mapping",
    ]

    def __init__(
        self,
        dataset_path: str,
        properties: DatasetProperties,
        label_conversion: Optional[bool] = False,
    ) -> None:
        self._properties: DatasetProperties = properties
        self._df: pd.DataFrame = self._load_df(dataset_path)
        self._transformations: List[Callable] = []
        self._label_mapping: Dict[Any, int] = None

        if label_conversion:
            self._df[self._properties.labels] = self._label_conversion()

    def _load_df(self, dataset_path: str) -> pd.DataFrame:
        if not dataset_path.lower().endswith(".csv"):
            raise NotImplementedError("Only CSV files are supported.")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"The dataset file was not found at the specified path: {dataset_path}"
            )

        logging.info(f"Reading dataset from: {dataset_path}...")
        return pd.read_csv(dataset_path)

    def _label_conversion(self) -> pd.DataFrame:
        logging.info("Mapping original class labels to numeric values...")
        self._label_mapping = {
            label: idx
            for idx, label in enumerate(self._df[self._properties.labels].unique())
        }
        return np.vectorize(self._label_mapping.get)(self._df[self._properties.labels])
    
    def add_step(self, order: int) -> Callable:
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            wrapper.order = order
            self._transformations.append(wrapper)
            logging.info(
                f"Added '{func.__name__}' to preprocessing pipeline with order {order}."
            )
            return wrapper
        return decorator

    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(f"Applying {len(self._transformations)} transformation...")
        for transform_function in sorted(
            self._transformations,
            key=lambda transform_function: transform_function.order,
        ):
            transform_function(self._df, self._properties)

        logging.info("Completed transformations.")
        return self._df[self._properties.features], self._df[self._properties.labels]


