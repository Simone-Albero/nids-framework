import numpy as np
import pandas as pd


def base_pre_processing(dataset: str) -> None:
    """Convert numerical columns to floats, and remove out of range values

    Args:
        The dataset to be processed.
    """
    threshold = 1000000
    for column_name, column_values in dataset.numerical_column_iterator():
        column_values[~np.isfinite(column_values)] = 0
        column_values[column_values < -threshold] = 0
        column_values[column_values > threshold] = 0
        column_values = column_values.astype("float32")
        dataset.update_column(column_name, column_values)
        # print(dataset._df[column_name])


def numerical_pre_processing(dataset: str) -> None:
    """Perform pre-processing operation on numerical values

    Args:
        The dataset to be processed.
    """
    # TODO: maybe 2 for lops are not necessary
    fits = {}
    for column_name, column_values in dataset.numerical_column_iterator(True):
        min_value = np.min(column_values)
        max_value = np.max(column_values)
        gap = max_value - min_value
        fits[column_name] = (min_value, gap)

    for column_name, column_values in dataset.numerical_column_iterator():
        min_value, gap = fits[column_name]

        if gap == 0:
            dataset.update_column(
                column_name, np.zeros_like(column_values, dtype="float32")
            )
        else:
            column_values -= min_value
            column_values = np.log(column_values + 1)
            column_values *= 1.0 / np.log(gap + 1)
            dataset.update_column(column_name, column_values)

        # print(dataset._df[column_name])


def categorical_pre_processing(dataset: str) -> None:
    """Perform pre-processing operation on categorical values

    Args:
        The dataset to be processed.
    """
    fits = {}
    categorical_bound = 30
    for column_name, column_values in dataset.categorical_column_iterator(True):
        values, frequencies = np.unique(column_values, return_counts=True)
        values = list(
            sorted(zip(values, frequencies), key=lambda x: x[1], reverse=True)
        )
        fits[column_name] = [s[0] for s in values[:categorical_bound]]

    for column_name, column_values in dataset.categorical_column_iterator():
        bounded_feature_values = fits[column_name]

        feature_representation = np.zeros(len(column_values), dtype="uint32")
        for value, frequency in enumerate(bounded_feature_values):
            mask = column_values == value
            feature_representation[mask] = frequency + 1

        feature_encoding = pd.get_dummies(feature_representation, prefix=column_name)
        # print(feature_encoding.head)

        # TODO: maybe dataset properties update needed
        dataset.add_columns(feature_encoding)

        dataset.drop_column(column_name)
