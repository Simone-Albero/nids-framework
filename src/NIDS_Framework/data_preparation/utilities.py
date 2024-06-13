from typing import Dict, Any, Tuple
import logging

import pandas as pd
import numpy as np
import torch

from data_preparation.processor import DatasetProperties


def base_pre_processing(
    dataset: pd.DataFrame, properties: DatasetProperties, bound: int = 1000000
) -> None:
    logging.debug(f"Bounding numeric features to {bound}...")
    for col in properties.numeric_features:
        column_values = dataset[col]
        column_values[~np.isfinite(column_values)] = 0
        column_values[column_values < -bound] = 0
        column_values[column_values > bound] = 0
        dataset[col] = column_values.astype("float32")


def log_pre_processing(
    dataset: pd.DataFrame, properties: DatasetProperties, bound: int = 1000000
) -> None:
    logging.debug("Normalizing numeric features with Log...")
    for col in properties.numeric_features:
        column_values = dataset[col]
        min_value = np.min(column_values)
        max_value = np.max(column_values)
        gap = max_value - min_value

        if gap == 0:
            dataset[col] = np.zeros_like(column_values, dtype="float32")
        else:
            column_values -= min_value
            column_values = np.log(column_values + 1)
            column_values *= 1.0 / np.log(gap + 1)
            dataset[col] = column_values


def categorical_pre_processing(
    dataset: pd.DataFrame, properties: DatasetProperties, categorical_levels: int = 32
) -> None:
    logging.debug(
        f"Mapping {len(properties.categorical_features)} categorical features to {categorical_levels} numeric tags..."
    )

    for col in properties.categorical_features:
        unique_values = dataset[col].unique()[: (categorical_levels - 1)]
        value_map = {val: idx for idx, val in enumerate(unique_values)}

        dataset[col] = dataset[col].apply(lambda x: value_map.get(x, -1) + 1)


def multi_class_label_conversion(
    dataset: pd.DataFrame, properties: DatasetProperties
) -> None:
    logging.debug("Mapping class labels to numeric values...")
    mapping = {}
    reverse = {}

    for idx, label in enumerate(dataset[properties.labels].unique()):
        mapping[label] = idx
        reverse[idx] = label

    dataset[properties.labels] = dataset[properties.labels].replace(mapping)
    return reverse


def bynary_label_conversion(
    dataset: pd.DataFrame, properties: DatasetProperties
) -> None:
    logging.debug("Mapping class labels to numeric values...")
    dataset[properties.labels] = np.where(
        dataset[properties.labels] != properties.benign_label, 1, 0
    )


def bound_transformation(
    sample: Dict[str, Any], bound: int = 1000000
) -> Dict[str, Any]:
    features, stats = sample["data"], sample["stats"]
    features = torch.clamp(features, min=-bound, max=bound)
    stats["min"] = torch.clamp(stats["min"], min=-bound)
    stats["max"] = torch.clamp(stats["max"], max=bound)

    logging.debug(f"Bound transformation result:\n{features}")

    sample["data"] = features
    return sample


def log_transformation(sample: Dict[str, Any]) -> Dict[str, Any]:
    features, stats = sample["data"], sample["stats"]

    min_values = stats["min"]
    max_values = stats["max"]
    gaps = max_values - min_values

    mask = gaps != 0
    features = torch.where(
        mask, torch.log(features + 1) / torch.log(gaps + 1), torch.zeros_like(features)
    )
    features = torch.where(mask, features, torch.zeros_like(features))

    logging.debug(f"Log transformation result:\n{features}")

    sample["data"] = features
    return sample


def one_hot_encoding(
    sample: Dict[str, Any], categorical_levels: int = 32
) -> Dict[str, Any]:
    features = sample["data"]

    one_hot = torch.nn.functional.one_hot(features, num_classes=categorical_levels)
    logging.debug(f"One-hot transformation result:\n{one_hot}")

    if len(one_hot.shape) == 2:
        sample["data"] = one_hot.flatten()
    elif len(one_hot.shape) > 2:
        sample["data"] = one_hot.view(one_hot.size(0), -1)

    return sample
