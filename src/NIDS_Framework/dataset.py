from typing import List, Optional
import os

import pandas as pd # dependences: pandas?

from utilities import CacheUtils

class DatasetProperties(object):
    __slots__ = ['features', 'categorical_features', 'numerical_features', 'labels_column', 'benign_label']

    def  __init__(self, features: List[str], categorical_features: List[str], labels_column: Optional[str] = None, benign_label: Optional[str] = None) -> None:
        """
        Defines the format of specific dataset.

        Params:
            - features: fields to include as part of the training.
            - categorical_features: fields that should be treated as categorical.
            - numerical_features: fields that should be treated as numerical.
            - labels_column: fields treated as class attributes.
            - benign_label: label of benign traffic.
        """
        self.features: List[str] = features
        self.categorical_features: List[str] = categorical_features
        self.numerical_features: List[str] = [feature for feature in features if feature not in categorical_features]
        self.labels_column: Optional[str] = labels_column
        self.benign_label: Optional[str] = benign_label

class Dataset(object):
    __slots__ = ['cache_path', '_df', '_properties']

    def __init__(self, cache_path: str, dataset_path: Optional[str] = None, properties: Optional[DatasetProperties] = None) -> None:
        """
        An abstraction for datasets.

        Params:
            - cache_path: the path where the cache file should be stored.
            - dataset_path: the path to the dataset file (optional).
            - properties: properties of the dataset (optional).
        """
        self.cache_path: str = cache_path
        self._df: Optional[pd.DataFrame] = None
        self._properties: Optional[DatasetProperties] = None

        if CacheUtils.exists(cache_path):
            self._load_from_cache()
        
        elif dataset_path is not None:
            self._load_from_file(dataset_path, properties)
        
        else:
            raise Exception(f"Please provide either the cache path or the dataset path!")
    
    class ColumnIterator:
        def __init__(self, features: List[str], blacklist: Optional[List[str]] = None) -> None:
            self._features = features
            self._blacklist = blacklist or []
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            while self._index < len(self._features):
                column_name = self._features[self._index]
                self._index += 1
                if column_name not in self._blacklist:
                    return column_name
            raise StopIteration

    def _load_from_file(self, dataset_path: str, properties: DatasetProperties) -> None:
        """
        Load dataset from file and save to cache.

        Params:
            - dataset_path: the path to the dataset file.
            - properties: properties of the dataset.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        print(f"Reading dataset from path '{dataset_path}' ...")
        if not dataset_path.lower().endswith(".csv"):
            raise NotImplementedError("Only CSV files are supported.")
        
        self._df = pd.read_csv(dataset_path)
        self._properties = properties
        self._save_to_cache()

    def _save_to_cache(self) -> None:
        """Save DataFrame and properties to cache."""
        data = {'df': self._df, 'properties': self._properties}
        CacheUtils.write(self.cache_path, data)

    def _load_from_cache(self) -> None:
        """Load DataFrame and properties from cache."""
        data = CacheUtils.read(self.cache_path)
        self._df, self._properties = data['df'], data['properties']
    
    def column_iterator(self):
        """Iterate over columns of the DataFrame."""
        return self.ColumnIterator(self._properties.features)
    
    def numerical_column_iterator(self):
        """Iterate over numerical columns of the DataFrame."""
        return self.ColumnIterator(self._properties.features, self._properties.categorical_features)
    
    def categorical_column_iterator(self):
        """Iterate over categorical columns of the DataFrame."""
        return self.ColumnIterator(self._properties.features, self._properties.numerical_features)

