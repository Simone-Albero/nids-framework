from typing import Optional, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..properties import DatasetProperties

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    __slots__ = ["properties", "categorical_levels", "value_map"]

    def __init__(self, properties: DatasetProperties, categorical_levels: int = 32):
        super().__init__()
        self.properties = properties
        self.categorical_levels = categorical_levels
        self.value_map: Dict[str, Dict] = {}

    def fit(self, data: pd.DataFrame, target: Optional[pd.DataFrame] = None):
        self.value_map = {
            col: {
                val: rank + 1
                for rank, val in enumerate(
                    data[col].value_counts().index[:self.categorical_levels - 1]
                )
            }
            for col in self.properties.categorical_features
            if col in data
        }
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.value_map:
            raise ValueError("The encoder must be fitted before transforming data.")

        transformed_data = data.copy()
        for col in self.properties.categorical_features:
            if col in transformed_data:
                transformed_data[col] = transformed_data[col].map(self.value_map.get(col, {})).fillna(0).astype(int)
        return transformed_data

