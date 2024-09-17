import pandas as pd
import logging
import os

from rich.logging import RichHandler

def generate_train_test(train_fraction=0.85):
    dataset_name = "NF-UNSW-NB15-V2"
    df_path = "datasets/NF-UNSW-NB15-V2/NF-UNSW-NB15-V2.csv"
    output_dir = "datasets/NF-UNSW-NB15-V2"

    df = pd.read_csv(df_path)
    train_size = int(len(df) * train_fraction)
    
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"{dataset_name}_train.csv")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.csv")

    df_train.to_csv(train_path, index=False)
    logging.info(f"Train saved in: {train_path}")
    
    df_test.to_csv(test_path, index=False)
    logging.info(f"Test saved in: {test_path}")

if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    generate_train_test()