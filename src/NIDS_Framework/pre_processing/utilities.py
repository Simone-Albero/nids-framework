import numpy as np


def base_pre_processing(dataset: str) -> None:
    """Perform basic preprocessing operations on numerical features.

    Args:
        The dataset to be processed.
    """
    for column_name, column_values in dataset.numerical_column_iterator():
        column_values[~np.isfinite(column_values)] = 0
        column_values[column_values < -1000000] = 0
        column_values[column_values > 1000000] = 0
        column_values = column_values.astype("float32")
        dataset.update_column(column_name, column_values)
        print(dataset._df[column_name])
