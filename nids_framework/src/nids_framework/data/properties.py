import configparser
import logging
from typing import List, Optional

class DatasetProperties:

    __slots__ = [
        "features",
        "categorical_features",
        "numeric_features",
        "label",
        "benign_label",
    ]

    def __init__(
        self,
        features: List[str],
        categorical_features: List[str],
        label: Optional[str] = None,
        benign_label: Optional[str] = None,
    ) -> None:
        self.features = features
        self.categorical_features = categorical_features
        self.numeric_features: List[str] = [
            feature for feature in features if feature not in categorical_features
        ]
        self.label = label
        self.benign_label = benign_label

        logging.info(
            f"Dataset properties: Total features: {len(features)} (Numeric: {len(self.numeric_features)}, Categorical: {len(categorical_features)})\n"
        )


class NamedDatasetProperties:

    __slots__ = [
        "config_path",
        "SEPARATOR",
        "_config",
    ]

    def __init__(self, config_path: str, separator: str = ",") -> None:
        self.config_path = config_path
        self.SEPARATOR = separator
        self._config: configparser.ConfigParser = configparser.ConfigParser()
        self._config.read(config_path)

    def get_properties(self, name: str) -> DatasetProperties:
        if name in self._config:
            logging.info(f"Reading '{name}' from '{self.config_path}'.")
            spec = self._config[name]
            return DatasetProperties(
                features=spec["features"].split(self.SEPARATOR),
                categorical_features=spec["categorical_features"].split(self.SEPARATOR),
                label=spec.get("label"),
                benign_label=spec.get("benign_label"),
            )
        else:
            raise ValueError(f"Properties for {name} not found.")
