from typing import List, Optional, Callable, Dict, Any, Tuple
import logging

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
        "_transformations_mappings",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        properties: DatasetProperties,
    ) -> None:
        self._properties: DatasetProperties = properties
        self._df: pd.DataFrame = df
        self._transformations: List[Callable] = []
        self._transformations_mappings: Dict[str, Dict[int, Any]] = {}

    @property
    def transformations(self) -> List[Callable]:
        return self._transformations

    @transformations.setter
    def transformations(self, transformations: List[Callable]) -> None:
        self._transformations = sorted(transformations, key=lambda transform_function: transform_function.order)
    
    def fit(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logging.info(f"Applying {len(self._transformations)} transformation...")
        for transform_function in self._transformations:
            maping = transform_function(self._df, self._properties)
            if maping is not None: self._transformations_mappings[transform_function.__name__] = maping

        logging.info("Completed.")
        return self._df[self._properties.features], self._df[self._properties.labels]
    
    def decode(self, data: int, map_name: str):
        if data not in self._transformations_mappings[map_name]: raise ValueError(f"Value {data} not in {map_name}.")
        return self._transformations_mappings[map_name][data]


