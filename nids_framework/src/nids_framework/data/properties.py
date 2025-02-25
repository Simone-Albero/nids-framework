import logging
from typing import List, Optional

from .config_manager import ConfigManager

class DatasetProperties:

    __slots__ = [
        "train_path",
        "test_path",
        "features",
        "categorical_features",
        "numeric_features",
        "label",
        "benign_label",
    ]

    def __init__(
        self,
        train_path: str,
        test_path: str,
        features: List[str],
        categorical_features: List[str],
        label: Optional[str] = None,
        benign_label: Optional[str] = None,
    ) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.features = features
        self.categorical_features = categorical_features
        self.numeric_features: List[str] = [
            feature for feature in features if feature not in categorical_features
        ]
        self.label = label
        self.benign_label = benign_label

        logging.info(
            f"DatasetProperties initialized: {len(features)} features "
            f"(Numeric: {len(self.numeric_features)}, Categorical: {len(self.categorical_features)})"
        )

    @classmethod
    def from_config(cls, config_path: str) -> "DatasetProperties":
        ConfigManager.load_config(config_path)

        train_path = ConfigManager.get_value('dataset', 'train_path')
        test_path = ConfigManager.get_value('dataset', 'test_path')
        features = ConfigManager.get_value('dataset', 'features', [])
        categorical_features = ConfigManager.get_value('dataset', 'categorical_features', [])
        label = ConfigManager.get_value('dataset', 'label')
        benign_label = ConfigManager.get_value('dataset', 'benign_label')

        return cls(
            train_path=train_path,
            test_path=test_path,
            features=features,
            categorical_features=categorical_features,
            label=label,
            benign_label=benign_label,
        )