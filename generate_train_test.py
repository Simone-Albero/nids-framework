import pandas as pd
import logging
import os

from rich.logging import RichHandler
import tqdm

import os
import pandas as pd
import logging

def generate_train_test(train_fraction=0.8, dataset_name="NF-UNSW-NB15-V2"):
    df_path = f"datasets/{dataset_name}/{dataset_name}.csv"
    output_dir = f"datasets/{dataset_name}"
    df = pd.read_csv(df_path)
    
    train_size = int(len(df) * train_fraction)
    
    df_train = df.head(train_size)
    df_test = df.tail(len(df) - train_size)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"{dataset_name}-Train.csv")
    test_path = os.path.join(output_dir, f"{dataset_name}-Test.csv")

    df_train.to_csv(train_path, index=False)
    logging.info(f"Training set saved to: {train_path}")
    
    df_test.to_csv(test_path, index=False)
    logging.info(f"Testing set saved to: {test_path}")


if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    generate_train_test()