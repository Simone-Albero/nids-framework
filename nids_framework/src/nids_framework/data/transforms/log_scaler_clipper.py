from typing import Optional, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..properties import DatasetProperties

class LogScalerClipper(BaseEstimator, TransformerMixin):
    __slots__ = ["properties", "min_values", "max_values", "border"]

    def __init__(self, properties: DatasetProperties, border: int = 100):
        super().__init__()
        self.properties = properties
        self.border = border
        self.min_values: Optional[pd.Series] = None
        self.max_values: Optional[pd.Series] = None

    def fit(self, data: pd.DataFrame, target: Optional[pd.DataFrame] = None):
        self.min_values = data.min().clip(lower=-self.border)
        self.max_values = data.max().clip(upper=self.border)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.min_values is None or self.max_values is None:
            raise ValueError("The scaler must be fitted before transforming data.")
        
        clipped_data = data[self.properties.numeric_features].apply(
            lambda x: np.clip(
                x.fillna(0).replace([np.inf, -np.inf], 0), -self.border, self.border
            ).astype(np.float32)
        )
        
        gaps = np.maximum(self.max_values - self.min_values, 1e-9)
        scaled_data = np.maximum(clipped_data - self.min_values, 0)
        transformed_data = np.log1p(scaled_data) / np.log1p(gaps)
        
        return transformed_data

