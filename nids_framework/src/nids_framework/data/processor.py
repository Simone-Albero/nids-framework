import logging

import pandas as pd

from .properties import DatasetProperties


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
        self._properties = properties
        self._df = df
        self._transformations: list[callable] = []

    @property
    def transformations(self) -> list[callable]:
        return self._transformations

    @transformations.setter
    def transformations(self, transformations: list[callable]) -> None:
        self._transformations = sorted(
            transformations, key=lambda transform_function: transform_function.order
        )

    def apply(self) -> None:
        logging.info(f"Applying {len(self._transformations)} transformation...")
        for transform_function in self._transformations:
            transform_function(self._df, self._properties)

        logging.info("Completed.\n")

    def build(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._df[self._properties.features], self._df[self._properties.labels]
