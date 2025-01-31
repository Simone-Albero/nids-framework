import pandas as pd
import logging
import os

from rich.logging import RichHandler
import tqdm

import os
import pandas as pd
import logging

def generate_train_test(train_fraction=0.8, dataset_name="NF-UNSW-NB15-V2", df_path="datasets/NF-UNSW-NB15-V2/NF-UNSW-NB15-V2.csv", output_dir="datasets/NF-UNSW-NB15-V2"):
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


def generate_custom():
    file_path = "datasets/NF-ToN-IoT-V2/NF-ToN-IoT-V2-Test.csv"

    df = pd.read_csv(file_path)
    custom_df = pd.DataFrame()

    attacks = df["Attack"].unique()
    attacks = attacks[(attacks != "Benign")]
    #attacks = ["DoS", "Reconnaissance", "Fuzzers"]
    benign_rows = df[df["Attack"] == "Benign"]

    added_indices = set()
    WINDOW = 10 
    ATTACK_SAMPLES = 500 

    for label in tqdm.tqdm(attacks):
        attack_mask = df[df["Attack"] == label]
        attack_mask = attack_mask.iloc[:ATTACK_SAMPLES, :]

        for idx in attack_mask.index:
            custom_df = pd.concat((custom_df, df.loc[[idx]]), axis=0)
            added_indices.add(idx)

            before_background = benign_rows.loc[max(0, idx - WINDOW) : idx - 1]
            after_background = benign_rows.loc[idx + 1 : idx + WINDOW]

            for bg_idx in before_background.index:
                if bg_idx not in added_indices:
                    custom_df = pd.concat(
                        (custom_df, benign_rows.loc[[bg_idx]]), axis=0
                    )
                    added_indices.add(bg_idx)

            for bg_idx in after_background.index:
                if bg_idx not in added_indices:
                    custom_df = pd.concat(
                        (custom_df, benign_rows.loc[[bg_idx]]), axis=0
                    )
                    added_indices.add(bg_idx)

    custom_df = custom_df.sort_index()
    print(custom_df["Attack"].value_counts())

    output_path = "datasets/NF-ToN-IoT-V2/NF-ToN-IoT-V2-Balanced-Test.csv"
    custom_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    generate_train_test()