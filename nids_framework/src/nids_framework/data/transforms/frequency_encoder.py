from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..properties import DatasetProperties


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    __slots__ = ["properties", "categorical_levels", "value_map"]

    def __init__(self, properties: DatasetProperties, categorical_levels: int = 32):
        super().__init__()
        self.properties = properties
        self.categorical_levels = categorical_levels
        self.value_map: Optional[dict[str, dict]] = None

    def fit(self, data: pd.DataFrame, target: Optional[pd.DataFrame] = None):
        self.value_map = {
            col: {
                val: rank + 1
                for rank, val in enumerate(
                    data[col].value_counts().index[: (self.categorical_levels - 1)]
                )
            }
            for col in self.properties.categorical_features
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.value_map is None:
            raise ValueError("The encoder must be fitted before transforming data.")
        return data[self.properties.categorical_features].apply(
            lambda col: col.map(self.value_map[col.name]).fillna(0).astype(int)
        )
