from typing import Optional, Dict, Any
import logging

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

from ..properties import DatasetProperties

class LabelEncoder(BaseEstimator, TransformerMixin):
    __slots__ = ["properties", "mapping"]

    def __init__(self, properties: DatasetProperties, mapping: Dict[Any, Any]):
        super().__init__()
        self.properties = properties
        self.mapping = mapping

    def fit(self, data: pd.DataFrame, target: Optional[pd.DataFrame] = None):
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(lambda col: col.map(self.mapping).fillna(-1).astype(int))

