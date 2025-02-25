from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..properties import DatasetProperties


class Clipper(BaseEstimator, TransformerMixin):
    __slots__ = ["properties", "border"]

    def __init__(self, properties: DatasetProperties, border: int = 100):
        super().__init__()
        self.properties = properties
        self.border = border

    def fit(self, data: pd.DataFrame, target: Optional[pd.DataFrame] = None):
        return self  # No fitting needed

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[self.properties.numeric_features].apply(
            lambda x: np.clip(
                x.fillna(0).replace([np.inf, -np.inf], 0), -self.border, self.border
            ).astype(np.float32)
        )
