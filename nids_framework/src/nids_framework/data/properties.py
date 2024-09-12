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
        features: list[str],
        categorical_features: list[str],
        labels: list[str] = None,
        benign_label: str = None,
    ) -> None:
        self.features = features
        self.categorical_features = categorical_features
        self.numeric_features: list[str] = [
            feature for feature in features if feature not in categorical_features
        ]
        self.labels = labels
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

    def __init__(self, config_path: str, saparator: str = ","):
        self.config_path = config_path
        self.SEPARATOR = saparator
        self._config: configparser.ConfigParser = configparser.ConfigParser()
        self._config.read(config_path)

    def get_properties(self, name: str):
        if name in self._config:
            logging.info(f"Reading '{name}' from '{self.config_path}'.")
            spec = self._config[name]
            return DatasetProperties(
                features=spec["features"].split(self.SEPARATOR),
                categorical_features=spec["categorical_features"].split(self.SEPARATOR),
                labels=spec["labels"],
                benign_label=spec["benign_label"],
            )
        else:
            raise ValueError(f"Properties for {name} not found.")
