from typing import Dict, Any
import logging

import pandas as pd
import numpy as np
import torch

from data.processor import DatasetProperties

def base_pre_processing(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    train_mask: pd.Series,
    bound: int,
) -> None:
    logging.debug(f"Bounding numeric features to {bound}...")
    for col in properties.numeric_features:
        column_values = dataset[col].values
        column_values[~np.isfinite(column_values)] = 0
        column_values[column_values < -bound] = 0
        column_values[column_values > bound] = 0
        dataset[col] = column_values.astype("float32")


def log_pre_processing(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    train_mask: pd.Series,
) -> None:
    logging.debug("Normalizing numeric features with Log...")
    for col in properties.numeric_features:
        column_values = dataset[col].values
        train_values = dataset[train_mask][col].values
        min_value = np.min(train_values)
        max_value = np.max(train_values)
        gap = max_value - min_value

        if gap == 0:
            dataset[col] = np.zeros_like(column_values, dtype="float32")
        else:
            column_values -= min_value
            column_values = np.log(column_values + 1)
            column_values *= 1.0 / np.log(gap + 1)
            dataset[col] = column_values


def categorical_pre_processing(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    train_mask: pd.Series,
    categorical_levels: int,
) -> None:
    logging.debug(
        f"Mapping {len(properties.categorical_features)} categorical features to {categorical_levels} numeric tags..."
    )

    for col in properties.categorical_features:
        unique_values = (
            dataset[train_mask][col].value_counts().index[: (categorical_levels - 1)]
        )
        value_map = {val: idx for idx, val in enumerate(unique_values)}

        dataset[col] = dataset[col].apply(lambda x: value_map.get(x, -1) + 1)


def multi_class_label_conversion(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    train_mask: pd.Series,
) -> Dict[int, Any]:
    logging.debug("Mapping class labels to numeric values...")
    mapping = {}
    reverse = {}

    for idx, label in enumerate(dataset[properties.labels].unique()):
        mapping[label] = idx
        reverse[idx] = label

    dataset[properties.labels] = dataset[properties.labels].apply(
        lambda x: mapping.get(x)
    )
    return reverse


def bynary_label_conversion(
    dataset: pd.DataFrame,
    properties: DatasetProperties,
    train_mask: pd.Series,
) -> None:
    logging.debug("Mapping class labels to numeric values...")
    dataset[properties.labels] = ~(
        dataset[properties.labels].astype("str") == str(properties.benign_label)
    )

def one_hot_encoding(sample: Dict[str, Any], levels: int) -> Dict[str, Any]:
    features = sample["data"]
    one_hot = torch.nn.functional.one_hot(features, num_classes=levels)

    if len(one_hot.shape) == 2:
        sample["data"] = one_hot.flatten()
    elif len(one_hot.shape) > 2:
        sample["data"] = one_hot.view(one_hot.size(0), -1)

    return sample

def log_transformation(sample: Dict[str, Any], min_values: pd.Series, max_values: pd.Series) -> Dict[str, Any]:
    features = sample["data"]
    gaps = max_values - min_values

    mask = gaps != 0
    features = torch.where(
        mask, torch.log(features + 1) / torch.log(gaps + 1), torch.zeros_like(features)
    )
    features = torch.where(mask, features, torch.zeros_like(features))
    sample["data"] = features
    return sample

def categorical_value_encoding(sample: Dict[str, Any], categorical_levels: pd.DataFrame, categorical_bound: int) -> Dict[str, Any]:
    features = sample["data"]
    categorical_levels = categorical_levels[:, :(categorical_bound - 1)]

    value_encoding = torch.zeros_like(features)
    for col_idx, col in enumerate(features.t()):
        for row_idx, val in enumerate(col):
            mask = (categorical_levels[:, col_idx] == val).nonzero()
            if mask.numel() > 0:
                value_encoding[row_idx, col_idx] = mask.item() + 1

    sample["data"] = value_encoding
    return sample
