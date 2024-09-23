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


def base_pre_processing_row(
    row: pd.Series,
    properties: DatasetProperties,
    bound: int,
) -> pd.Series:
    logging.debug(f"Bounding numeric features to {bound} for a single row...")

    row_copy = row.copy()

    for col in properties.numeric_features:
        if col in row_copy:
            value = row_copy[col]
            if pd.isna(value) or value == np.inf or value == -np.inf:
                row_copy[col] = 0
            else:
                row_copy[col] = np.clip(value, -bound, bound).astype("float32")

    return row_copy


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


def log_pre_processing_row(
    row: pd.Series,
    properties: DatasetProperties,
    min_values: dict[str, float],
    max_values: dict[str, float],
) -> pd.Series:
    logging.debug("Normalizing numeric features with Log for a single row...")

    row_copy = row.copy()

    for col in properties.numeric_features:
        if col in row_copy:
            min_value = min_values[col]
            max_value = max_values[col]
            gap = max_value - min_value

            if gap == 0:
                row_copy[col] = 0.0
            else:
                value = row_copy[col]
                if pd.isna(value) or value == np.inf or value == -np.inf:
                    row_copy[col] = 0.0
                else:
                    value -= min_value
                    value = np.maximum(value, 0)
                    value = np.log(value + 1)
                    normalization_factor = 1.0 / np.log(gap + 1)
                    row_copy[col] = value * normalization_factor

    return row_copy


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


def categorical_pre_processing_row(
    row: pd.Series,
    properties: DatasetProperties,
    unique_values: dict[str, list[any]],
    categorical_levels: int,
) -> pd.Series:
    logging.debug(
        f"Mapping {len(properties.categorical_features)} categorical features to {categorical_levels} numeric tags for a single row..."
    )

    row_copy = row.copy()

    for col in properties.categorical_features:
        if col in row_copy:
            value_map = {val: idx + 1 for idx, val in enumerate(unique_values[col])}
            row_copy[col] = value_map.get(row_copy[col], 0)

    return row_copy


def binary_benign_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")

    df_copy = df.copy()

    df_copy[properties.labels] = ~(
        df_copy[properties.labels].astype("str") == properties.benign_label
    )

    return df_copy

def binary_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")

    df_copy = df.copy()

    df_copy[properties.labels] = (
        df_copy[properties.labels].astype("str") == properties.benign_label
    )

    return df_copy


def binary_label_conversion_row(
    row: pd.Series,
    properties: DatasetProperties,
) -> pd.Series:
    logging.debug("Converting class label to numeric value for a single row...")

    row_copy = row.copy()

    row_copy[properties.labels] = not (
        str(row_copy[properties.labels]) == properties.benign_label
    )

    return row_copy


def multi_class_label_conversion(
    df: pd.DataFrame,
    properties: DatasetProperties,
    mapping: dict[any, int],
) -> pd.DataFrame:
    logging.debug("Converting class labels to numeric values...")

    df_copy = df.copy()
    df_copy[properties.labels] = (
        df_copy[properties.labels].map(mapping).fillna(-1).astype(int)
    )

    return df_copy


def multi_class_label_conversion_row(
    row: pd.Series,
    properties: DatasetProperties,
    mapping: dict[any, int],
) -> pd.Series:
    logging.debug("Converting class label to numeric value for a single row...")

    row_copy = row.copy()

    row_copy[properties.labels] = mapping.get(row_copy[properties.labels], -1)

    return row_copy


def split_data_for_torch(
    df: pd.DataFrame,
    properties: DatasetProperties,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.debug("Extracting features and labels for Torch...")

    features_df = df[properties.features]
    labels_df = df[properties.labels]

    return features_df, labels_df


def split_data_for_torch_row(
    row: pd.Series,
    properties: DatasetProperties,
) -> tuple[pd.Series, pd.Series]:
    logging.debug("Extracting features and labels for Torch for a single row...")

    features = row[properties.features]
    labels = row[properties.labels]

    return features, labels


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
    tensor: torch.Tensor, mask_prob: float = 0.1, mask_value: any = 0.0
) -> torch.Tensor:
    mask = torch.rand(tensor.shape, device=tensor.device) < mask_prob

    return torch.where(mask, torch.tensor(mask_value, device=tensor.device), tensor)
