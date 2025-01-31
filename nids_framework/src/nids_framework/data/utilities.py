import logging

import pandas as pd
import numpy as np
import torch

from .properties import DatasetProperties
from typing import Dict, List, Tuple, Any


def min_max_values(
    df: pd.DataFrame, properties: DatasetProperties, bound: int = np.inf
) -> Tuple[Dict[str, float], Dict[str, float]]:
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
) -> Dict[str, List[Any]]:
    unique_values = {}
    levels = max(1, categorical_levels - 1)

    for col in properties.categorical_features:
        uniques = df[col].value_counts().index[: levels].tolist()
        unique_values[col] = uniques

    return unique_values


def labels_mapping(
    df: pd.DataFrame, properties: DatasetProperties
) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    logging.debug("Mapping class labels to numeric values...")
    mapping = {}
    reverse = {}

    for idx, label in enumerate(df[properties.label].unique()):
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
    df_copy[properties.numeric_features] = df_copy[properties.numeric_features].apply(
        lambda x: np.clip(x.fillna(0).replace([np.inf, -np.inf], 0), -bound, bound).astype("float32")
    )
    return df_copy


def log_pre_processing(
    df: pd.DataFrame,
    properties: DatasetProperties,
    min_values: Dict[str, float],
    max_values: Dict[str, float],
) -> pd.DataFrame:
    logging.debug("Normalizing numeric features with Log...")

    df_copy = df.copy()

    for col in properties.numeric_features:
        min_value = min_values[col]
        max_value = max_values[col]
        gap = max(max_value - min_value, 1e-9)

        df_copy[col] = df_copy[col].replace([np.inf, -np.inf, np.nan], 0)
        df_copy[col] = np.maximum(df_copy[col] - min_value, 0)
        df_copy[col] = np.log1p(df_copy[col]) / np.log1p(gap)
    return df_copy


def categorical_pre_processing(
    df: pd.DataFrame,
    properties: DatasetProperties,
    unique_values: Dict[str, List[Any]],
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


def binary_benign_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")

    df_copy = df.copy()
    df_copy[properties.label] = (df_copy[properties.label].astype(str) != properties.benign_label).astype(int)
    
    return df_copy


def binary_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")

    df_copy = df.copy()
    df_copy[properties.label] = (df_copy[properties.label].astype(str) == properties.benign_label).astype(int)

    return df_copy


def multi_class_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
    mapping: Dict[Any, int],
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")

    df_copy = df.copy()
    df_copy[properties.label] = df_copy[properties.label].map(mapping).fillna(-1).astype(int)

    return df_copy


def split_data_for_torch(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.debug("Extracting features and labels for Torch...")

    return df[properties.features], df[properties.label]


def log_transformation(
    tensor: torch.Tensor, min_values: torch.Tensor, max_values: torch.Tensor
) -> torch.Tensor:
    gaps = max_values - min_values

    mask = gaps != 0

    transformed_tensor = torch.zeros_like(tensor)
    transformed_tensor[mask] = torch.log(tensor[mask] + 1) / torch.log(gaps[mask] + 1)

    return transformed_tensor


def categorical_value_encoding(
    tensor: torch.Tensor, categorical_levels: torch.Tensor, categorical_bound: int
) -> torch.Tensor:
    categorical_levels = categorical_levels[:, : (categorical_bound - 1)]

    transformed_tensor = torch.zeros_like(tensor, dtype=torch.long)
    for col_idx in range(tensor.size(1)):
        col_values = tensor[:, col_idx].unsqueeze(1)  # Shape (N, 1)
        cat_col_levels = categorical_levels[:, col_idx].unsqueeze(0)  # Shape (1, L)
        mask = col_values == cat_col_levels  # Shape (N, L)

        encoded_indices = (
            mask.nonzero(as_tuple=True)[1].reshape(-1, tensor.size(0)).t() + 1
        )
        transformed_tensor[:, col_idx] = encoded_indices.squeeze()

    return transformed_tensor


def one_hot_encoding(tensor: torch.Tensor, levels: int) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(tensor, num_classes=levels)

    if len(one_hot.shape) == 2:
        return one_hot.flatten()
    elif len(one_hot.shape) > 2:
        return one_hot.view(one_hot.size(0), -1)
    

def mask_features(
    tensor: torch.Tensor, mask_prob: float = 0.1, mask_value: Any = 0.0
) -> torch.Tensor:

    last_row = tensor[-1, :]
    mask = torch.rand(last_row.shape, device=tensor.device) < mask_prob
    
    tensor[-1, :] = torch.where(mask, torch.tensor(mask_value, device=tensor.device, dtype=tensor.dtype), last_row)
    
    return tensor
