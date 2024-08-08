from typing import List, Optional
import logging

import configparser

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
            f"Dataset properties: Total features: {len(features)} (Numeric: {len(self.numeric_features)}, Categorical: {len(categorical_features)})\n"
        )

class NamedDatasetProperties:

    __slots__ = [
        "config_path",
        "config",
        "SEPARATOR",
    ]

    def __init__(self, config_path: str, saparator: Optional[str] = ","):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.SEPARATOR = saparator
    
    def get_properties(self, name: str):
        if name in self.config:
            logging.info(f"Reading '{name}' from '{self.config_path}'.")
            spec = self.config[name]
            return DatasetProperties(
                features=spec['features'].split(self.SEPARATOR),
                categorical_features=spec['categorical_features'].split(self.SEPARATOR),
                labels=spec['labels'],
                benign_label=spec['benign_label']
            )
        else:
            raise ValueError(f"Properties for {name} not found.")
 