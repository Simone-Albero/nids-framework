from typing import List, Optional, Callable, Dict, Any, Tuple
import logging

import pandas as pd
import numpy as np
from numpy.typing import NDArray

from data.properties import DatasetProperties

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
            transform_function(self._df, self._properties)

        logging.info("Completed.\n")

    def build(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._df[self._properties.features], self._df[self._properties.labels]

