from typing import Optional

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..properties import DatasetProperties


class LogScaler(BaseEstimator, TransformerMixin):
    __slots__ = ["properties", "min_values", "max_values"]

    def __init__(self, properties: DatasetProperties):
        super().__init__()
        self.properties = properties
        self.min_values: Optional[pd.Series] = None
        self.max_values: Optional[pd.Series] = None

    def fit(self, data: pd.DataFrame, target: Optional[pd.DataFrame] = None):
        self.min_values = data.min()
        self.max_values = data.max()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.min_values is None or self.max_values is None:
            raise ValueError("The scaler must be fitted before transforming data.")
        gaps = np.maximum(self.max_values - self.min_values, 1e-9)
        scaled_data = np.maximum(data - self.min_values, 0)
        transformed_data = np.log1p(scaled_data) / np.log1p(gaps)

        return transformed_data
