import logging

import pandas as pd
import numpy as np
import torch

from .properties import DatasetProperties
from typing import Dict, List, Tuple, Any


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
            row_copy[col] = np.clip(0 if pd.isna(value) or value in [np.inf, -np.inf] else value, -bound, bound).astype("float32")
    return row_copy


def log_pre_processing_row(
    row: pd.Series,
    properties: DatasetProperties,
    min_values: Dict[str, float],
    max_values: Dict[str, float],
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


def categorical_pre_processing_row(
    row: pd.Series,
    properties: DatasetProperties,
    unique_values: Dict[str, List[Any]],
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


def binary_benign_label_conversion_row(
    row: pd.Series,
    properties: DatasetProperties,
) -> pd.Series:
    logging.debug("Converting class label to numeric value for a single row...")

    row_copy = row.copy()

    row_copy[properties.label] = not (
        str(row_copy[properties.label]) == properties.benign_label
    )

    return row_copy


def binary_label_conversion_row(
    row: pd.Series,
    properties: DatasetProperties,
) -> pd.Series:
    logging.debug("Converting class label to numeric value for a single row...")

    row_copy = row.copy()

    row_copy[properties.label] = (
        str(row_copy[properties.label]) == properties.benign_label
    )

    return row_copy


def multi_class_label_conversion_row(
    row: pd.Series,
    properties: DatasetProperties,
    mapping: Dict[Any, int],
) -> pd.Series:
    logging.debug("Converting class label to numeric value for a single row...")

    row_copy = row.copy()

    row_copy[properties.label] = mapping.get(row_copy[properties.label], -1)

    return row_copy


def split_data_for_torch_row(
    row: pd.Series,
    properties: DatasetProperties,
) -> Tuple[pd.Series, pd.Series]:
    logging.debug("Extracting features and labels for Torch for a single row...")

    features = row[properties.features]
    labels = row[properties.label]

    return features, labels