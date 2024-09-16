import logging

import pandas as pd
import numpy as np
import torch

from .properties import DatasetProperties


def min_max_values(
    df: pd.DataFrame, properties: DatasetProperties, bound: int = np.inf
) -> tuple[dict[str, float], dict[str, float]]:
    min_values = {}
    max_values = {}

    for col in properties.numeric_features:
        column_values = df[col].values
        min_value = np.min(column_values)
        max_value = np.max(column_values)

        min_values[col] = max(min_value, -bound)
        max_values[col] = min(max_value, bound)

    return min_values, max_values


def unique_values(
    df: pd.DataFrame, properties: DatasetProperties, categorical_levels: int
) -> dict[str, list[any]]:
    unique_values = {}

    for col in properties.categorical_features:
        uniques = df[col].value_counts().index[: (categorical_levels - 1)].tolist()
        unique_values[col] = uniques

    return unique_values


def labels_mapping(
    df: pd.DataFrame, properties: DatasetProperties
) -> tuple[dict[any, int], dict[int, any]]:
    logging.debug("Mapping class labels to numeric values...")
    mapping = {}
    reverse = {}

    for idx, label in enumerate(df[properties.labels].unique()):
        mapping[label] = idx
        reverse[idx] = label

    return mapping, reverse


def base_pre_processing(
    df: pd.DataFrame,
    properties: DatasetProperties,
    bound: int,
) -> pd.DataFrame:
    logging.debug(f"Bounding numeric features to {bound}...")

    df_copy = df.copy()
    
    for col in properties.numeric_features:
        column_values = df_copy[col]
        column_values.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        column_values.clip(lower=-bound, upper=bound, inplace=True)
        df_copy[col] = column_values.astype("float32")
    
    return df_copy


def log_pre_processing(
    df: pd.DataFrame,
    properties: DatasetProperties,
    min_values: dict[str, float],
    max_values: dict[str, float],
) -> pd.DataFrame:
    logging.debug("Normalizing numeric features with Log...")
    
    df_copy = df.copy()
    
    for col in properties.numeric_features:
        min_value = min_values[col]
        max_value = max_values[col]
        gap = max_value - min_value
        
        if gap == 0:
            df_copy[col] = 0.0
        else:
            column_values = df_copy[col]
            column_values.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
            column_values -= min_value
            column_values = np.maximum(column_values, 0)
            column_values = np.log(column_values + 1)
            normalization_factor = 1.0 / np.log(gap + 1)
            df_copy[col] = column_values * normalization_factor
    
    return df_copy


def categorical_pre_processing(
    df: pd.DataFrame,
    properties: DatasetProperties,
    unique_values: dict[str, list[any]],
    categorical_levels: int,
) -> pd.DataFrame:
    logging.debug(
        f"Mapping {len(properties.categorical_features)} categorical features to {categorical_levels} numeric tags..."
    )
    
    df_copy = df.copy()
    
    for col in properties.categorical_features:
        value_map = {val: idx + 1 for idx, val in enumerate(unique_values[col])}
        df_copy[col] = df_copy[col].map(value_map).fillna(0).astype(int)
    
    return df_copy


def binary_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")
    
    df_copy = df.copy()
    
    df_copy[properties.labels] = ~(
        df_copy[properties.labels].astype("str") == properties.benign_label
    )
    
    return df_copy


def multi_class_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
    mapping: dict[any, int],
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")
    
    df_copy = df.copy()
    df_copy[properties.labels] = df_copy[properties.labels].map(mapping).fillna(-1).astype(int)
    
    return df_copy

def split_data_for_torch(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.debug("Extracting features and labels for Torch...")

    features_df = df[properties.features]
    labels_df = df[properties.labels]
    
    return features_df, labels_df


def log_transformation(
    sample: dict[str, any], min_values: torch.Tensor, max_values: torch.Tensor
) -> dict[str, any]:
    features = sample["data"]
    gaps = max_values - min_values

    mask = gaps != 0

    features_transformed = torch.zeros_like(features)
    features_transformed[mask] = torch.log(features[mask] + 1) / torch.log(
        gaps[mask] + 1
    )

    sample["data"] = features_transformed
    return sample


def categorical_value_encoding(
    sample: dict[str, any], categorical_levels: torch.Tensor, categorical_bound: int
) -> dict[str, any]:
    features = sample["data"]
    categorical_levels = categorical_levels[:, : (categorical_bound - 1)]

    value_encoding = torch.zeros_like(features, dtype=torch.long)
    for col_idx in range(features.size(1)):
        col_values = features[:, col_idx].unsqueeze(1)  # Shape (N, 1)
        cat_col_levels = categorical_levels[:, col_idx].unsqueeze(0)  # Shape (1, L)
        mask = col_values == cat_col_levels  # Shape (N, L)

        encoded_indices = (
            mask.nonzero(as_tuple=True)[1].reshape(-1, features.size(0)).t() + 1
        )
        value_encoding[:, col_idx] = encoded_indices.squeeze()

    sample["data"] = value_encoding
    return sample


def one_hot_encoding(sample: dict[str, any], levels: int) -> dict[str, any]:
    features = sample["data"]
    one_hot = torch.nn.functional.one_hot(features, num_classes=levels)

    if len(one_hot.shape) == 2:
        sample["data"] = one_hot.flatten()
    elif len(one_hot.shape) > 2:
        sample["data"] = one_hot.view(one_hot.size(0), -1)

    return sample


def mask_features(
    sample: dict[str, any], mask_prob: float = 0.1, mask_value: any = 0.0
) -> dict[str, any]:
    data = sample["data"]
    mask = torch.rand(data.shape, device=data.device) < mask_prob

    data = torch.where(mask, torch.tensor(mask_value, device=data.device), data)
    sample["data"] = data
    return sample
