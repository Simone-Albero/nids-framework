import logging
import pickle

import pandas as pd
from rich.logging import RichHandler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np

def ms_stamp_update():
    dataset_path = "dataset/UGR16/custom/fixed_test.csv"
    df = pd.read_csv(dataset_path)

    df['te'] = pd.to_datetime(df['te'])
    df['td'] = df['td'] * 1000
    df['te'] = df['te'] + pd.to_timedelta(df['td'], unit='ms')
    df = df.sort_values(by='te')

    df.to_csv('dataset/UGR16/custom/ms_test.csv', index=False)

def fixed_windows_analisys():
    dataset_path = "dataset/UGR16/custom/ms_train.csv"
    df = pd.read_csv(dataset_path)
    df['te'] = pd.to_datetime(df['te'])

    WINDOW_SIZE = 10
    TOLERANCE = 0.2
    window_gaps = {}

    for i in tqdm(range(WINDOW_SIZE-1, len(df))):
        end = df.iloc[i]['te']
        start = df.iloc[i-WINDOW_SIZE+1]['te']
        gap = end - start
        gap = gap.total_seconds() * 1000

        found = False

        for key in window_gaps.keys():
            if abs(key - gap) <= gap * TOLERANCE:
                window_gaps[key] += 1
                found = True
                break

        if not found:
            window_gaps[gap] = 1

    with open('saves/window_gaps.pkl', 'wb') as file:
        pickle.dump(window_gaps, file)

    xs = sorted(window_gaps.keys())
    ys = [window_gaps[x] for x in xs]
    indices = list(range(len(xs)))
    
    plt.figure(figsize=(20, 10))
    plt.bar(indices, ys)
    plt.xlabel('Gap Size (ms)')
    plt.ylabel('Occurrences')
    plt.yscale('log')
    plt.title('Distribution of Gap Sizes')
    plt.xticks(indices, xs, rotation=90)
    plt.tight_layout()
    plt.show()

    plt.savefig('saves/window_gaps.png', bbox_inches='tight')
    plt.show()

def fixed_windows_dataset():
    DATASET_NAME = "ms_1_test"
    dataset_path = "dataset/UGR16/custom/" + DATASET_NAME + ".csv"
    df = pd.read_csv(dataset_path)
    df['te'] = pd.to_datetime(df['te'])

    WINDOW_SIZE_MS = pd.Timedelta(milliseconds=10)
    MAX_WINDOW = 64

    dfs = {2: [], 4: [], 8: [], 16: [], 32: [], 64: []}

    end_stamps = df['te'] + WINDOW_SIZE_MS

    for window_size in tqdm(dfs.keys()):
        valid_indices = np.where((df['te'].shift(-window_size + 1) <= end_stamps)[:len(df) - window_size + 1])[0] + window_size - 1
        dfs[window_size] = valid_indices.tolist()

    for window_size, df_list in dfs.items():
        if df_list:
            indices_df = pd.DataFrame(df_list, columns=['index'])
            indices_df.to_csv(f"dataset/UGR16/fixed/{DATASET_NAME}_{window_size}.csv", index=False)
            print(f"{window_size}: {len(df_list)}")   

def avg_durations():
    dataset_path = "dataset/UGR16/custom/ms_train.csv"
    df = pd.read_csv(dataset_path)
    df['te'] = pd.to_datetime(df['te'])

    TOLERANCE = 0.1

    duration_occurrences = {}

    for duration in tqdm(df['td']):
        found = False

        for key in duration_occurrences.keys():
            if abs(key - duration) <= duration * TOLERANCE:
                duration_occurrences[key] += 1
                found = True
                break

        if not found:
            duration_occurrences[duration] = 1

    with open('plots/duration_occurrences.pkl', 'wb') as file:
        pickle.dump(duration_occurrences, file)

    xs = sorted(duration_occurrences.keys())[:100]
    ys = [duration_occurrences[x] for x in xs]
    indices = list(range(len(xs)))

    plt.figure(figsize=(20, 10))
    plt.bar(indices, ys)
    plt.xlabel('Duration')
    plt.ylabel('Occurrences')
    plt.title('Distribution of Duration')
    plt.xticks(indices, xs, rotation=90)
    plt.tight_layout()

    plt.savefig('plots/duration_occurrences.png', bbox_inches='tight')
    plt.show()

def fragmented_test_set():
    DATASET_NAME = "ms_1_test"
    dataset_path = "dataset/UGR16/custom/" + DATASET_NAME + ".csv"
    df = pd.read_csv(dataset_path)
    df['te'] = pd.to_datetime(df['te'])

    WINDOW_SIZE_MS = pd.Timedelta(milliseconds=10)
    MAX_WINDOW = 64

    dfs = {2: [], 4: [], 8: [], 16: [], 32: [], 64: []}

    end_stamps = df['te'] + WINDOW_SIZE_MS
    assigned_indices = pd.Series(False, index=df.index)

    for window_size in tqdm([64, 32, 16, 8, 4, 2]):
        valid_indices = np.where((df['te'].shift(-window_size + 1) <= end_stamps)[:len(df) - window_size + 1])[0] + window_size - 1

        valid_indices = valid_indices[~assigned_indices.loc[valid_indices]]
        dfs[window_size] = valid_indices.tolist()

        for idx in valid_indices:
            assigned_indices.iloc[idx - window_size + 1:idx + 1] = True

    for window_size, df_list in dfs.items():
        if df_list:
            indices_df = pd.DataFrame(df_list, columns=['index'])
            indices_df.to_csv(f"dataset/UGR16/fragmented/{DATASET_NAME}_{window_size}.csv", index=False)
            print(f"{window_size}: {len(df_list)}")  



def main():
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )
    fixed_windows_dataset()
    fragmented_test_set()


if __name__ == "__main__":
    main()