from typing import Dict, Any, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np
import torch

from .processor import DatasetProperties

def min_max_values(dataset: pd.DataFrame, properties: DatasetProperties, bound: Optional[int] = np.inf) -> Tuple[Dict[str,float], Dict[str,float]]:
    min_values = {}
    max_values = {}
    
    for col in properties.numeric_features:
        column_values = dataset[col].values
        min_value = np.min(column_values)
        max_value = np.max(column_values)

        min_values[col] = max(min_value, -bound)
        max_values[col] = min(max_value, bound)

    return min_values, max_values


def unique_values(dataset: pd.DataFrame, properties: DatasetProperties, categorical_levels: int) -> Dict[str,List[Any]]:
    unique_values = {}

    for col in properties.categorical_features:
        uniques = dataset[col].value_counts().index[: (categorical_levels - 1)].tolist()
        unique_values[col] = uniques

    return unique_values

def labels_mapping(dataset: pd.DataFrame, properties: DatasetProperties) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    logging.debug("Mapping class labels to numeric values...")
    mapping = {}
    reverse = {}

    for idx, label in enumerate(dataset[properties.labels].unique()):
        mapping[label] = idx
        reverse[idx] = label

    return mapping, reverse

def base_pre_processing(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    bound: int,
) -> None:
    logging.debug(f"Bounding numeric features to {bound}...")
    for col in properties.numeric_features:
        column_values = dataset[col].values
        column_values[~np.isfinite(column_values)] = 0
        column_values[column_values < -bound] = -bound
        column_values[column_values > bound] = bound
        dataset[col] = column_values.astype("float32")


def log_pre_processing(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    min_values: Dict[str, float],
    max_values: Dict[str, float],
) -> None:
    logging.debug("Normalizing numeric features with Log...")
    for col in properties.numeric_features:
        column_values = dataset[col].values
        min_value , max_value = min_values[col], max_values[col]
        gap = max_value - min_value

        if gap == 0:
            dataset[col] = np.zeros_like(column_values, dtype="float32")
        else:
            column_values -= min_value
            column_values = np.maximum(column_values, 0) # maybe not fair
            column_values = np.log(column_values + 1)
            column_values *= 1.0 / np.log(gap + 1)
            dataset[col] = column_values


def categorical_pre_processing(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    unique_values: Dict[str, List[Any]],
    categorical_levels: int,
) -> None:
    logging.debug(
        f"Mapping {len(properties.categorical_features)} categorical features to {categorical_levels} numeric tags..."
    )

    for col in properties.categorical_features:
        value_map = {val: idx for idx, val in enumerate(unique_values[col])}

        dataset[col] = dataset[col].apply(lambda x: value_map.get(x, -1) + 1)


def bynary_label_conversion(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
) -> None:
    logging.debug("Converting class labels to numeric values...")
    dataset[properties.labels] = ~(
        dataset[properties.labels].astype("str") == properties.benign_label
    )

def multi_class_label_conversion(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    mapping: Dict[Any, int],
) -> None:
    logging.debug("Converting class labels to numeric values...")
    dataset[properties.labels] = dataset[properties.labels].apply(lambda x: mapping.get(x))

def log_transformation(sample: Dict[str, Any], min_values: torch.Tensor, max_values: torch.Tensor) -> Dict[str, Any]:
    features = sample["data"]
    gaps = max_values - min_values

    mask = gaps != 0

    features_transformed = torch.zeros_like(features)
    features_transformed[mask] = torch.log(features[mask] + 1) / torch.log(gaps[mask] + 1)

    sample["data"] = features_transformed
    return sample

def categorical_value_encoding(sample: Dict[str, Any], categorical_levels: torch.Tensor, categorical_bound: int) -> Dict[str, Any]:
    features = sample["data"]
    categorical_levels = categorical_levels[:, :(categorical_bound - 1)]


    value_encoding = torch.zeros_like(features, dtype=torch.long)
    for col_idx in range(features.size(1)):
        col_values = features[:, col_idx].unsqueeze(1)  # Shape (N, 1)
        cat_col_levels = categorical_levels[:, col_idx].unsqueeze(0)  # Shape (1, L)
        mask = col_values == cat_col_levels  # Shape (N, L)

        encoded_indices = mask.nonzero(as_tuple=True)[1].reshape(-1, features.size(0)).t() + 1
        value_encoding[:, col_idx] = encoded_indices.squeeze()

    sample["data"] = value_encoding
    return sample

def one_hot_encoding(sample: Dict[str, Any], levels: int) -> Dict[str, Any]:
    features = sample["data"]
    one_hot = torch.nn.functional.one_hot(features, num_classes=levels)

    if len(one_hot.shape) == 2:
        sample["data"] = one_hot.flatten()
    elif len(one_hot.shape) > 2:
        sample["data"] = one_hot.view(one_hot.size(0), -1)

    return sample

def mask_features(sample: Dict[str, Any], mask_prob: Optional[float] = 0.1, mask_value: Optional[Any] = 0.0) -> Dict[str, Any]:
    data = sample["data"]
    mask = torch.rand(data.shape, device=data.device) < mask_prob

    data = torch.where(mask, torch.tensor(mask_value, device=data.device), data)
    sample["data"] = data
    return sample