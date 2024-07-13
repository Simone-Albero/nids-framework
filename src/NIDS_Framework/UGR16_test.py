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


def ms_stamp_update():
    dataset_path = "dataset/UGR16/custom/fixed_test.csv"
    df = pd.read_csv(dataset_path)

    df['te'] = pd.to_datetime(df['te'])
    df['td'] = df['td'] * 1000
    df['te'] = df['te'] + pd.to_timedelta(df['td'], unit='ms')
    df = df.sort_values(by='te')

    df.to_csv('dataset/UGR16/custom/ms_test.csv', index=False)

def fixed_windows_dataset():
    dataset_path = "dataset/UGR16/custom/ms_test.csv"
    df = pd.read_csv(dataset_path)
    df['te'] = pd.to_datetime(df['te'])

    WINDOW_SIZE_MS = pd.Timedelta(milliseconds=10)
    MAX_WINDOW = 64
    last_stamp = df.iloc[-1]['te']
    curr_stamp = df.iloc[0]['te']

    dfs = {1: [], 2: [], 4: [], 8: [], 16: [], 32: [], 64: []}

    i = 0
    while curr_stamp < last_stamp:
        curr_window = MAX_WINDOW

        if i + curr_window >= len(df):
            break

        while True:
            if (df.iloc[i + curr_window]['te'] - curr_stamp <= WINDOW_SIZE_MS) or curr_window == 1:
                break
            else: 
                curr_window //= 2
        
        dfs[curr_window].append(df.iloc[i:i + curr_window])
        i += curr_window
        # i += 1
        curr_stamp = df.iloc[i]['te']
        print(f"Progress: {curr_stamp} / {last_stamp} : [{i}/{len(df)}]", end='\r')

    for window_size, df_list in dfs.items():
        if df_list:
            pd.concat(df_list).to_csv(f"dataset/UGR16/custom/{window_size}_test.csv", index=False)
            print(f"{window_size}:{len(df_list)}")

    print(f"Tot: {len(df)}")

def proportion_analysis():
    dataset_path = "dataset/UGR16/fixed/8_fixed.csv"
    df = pd.read_csv(dataset_path)

    attack_mask = df["label"] == "dos"
    attack_num = attack_mask.sum()
    proportion = attack_num / len(df)
    print(proportion)

def proportion_fix():
    P = 0.0185
    WINDOW_SIZE = 32

    dataset_path = "dataset/UGR16/custom/32_test.csv"
    df = pd.read_csv(dataset_path)

    print("Generating blocks...")
    positive_blocks = []
    negative_blocks = []
    for i in tqdm(range(0, len(df), WINDOW_SIZE)):
        block = df.iloc[i:i + WINDOW_SIZE]
        if block.iloc[-1]["label"] == "dos":
            positive_blocks.append(block)
        else:
            negative_blocks.append(block)

    random.shuffle(negative_blocks)

    num_ones = len(positive_blocks)
    num_negatives_to_keep = int(num_ones / P)
    print(num_ones, num_negatives_to_keep)

    negative_blocks = negative_blocks[:num_negatives_to_keep]
    
    print("Concatenating blocks...")
    df_positive = pd.concat(positive_blocks)
    df_negative = pd.concat(negative_blocks)

    df_fixed = pd.concat([df_positive, df_negative]).reset_index(drop=True)
    print(len(df_fixed))
    output_path = f"dataset/UGR16/fixed/{WINDOW_SIZE}_fixed.csv"
    df_fixed.to_csv(output_path, index=False)



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

def fixed_gap_analisys():
    dataset_path = "dataset/UGR16/custom/ms_train.csv"
    df = pd.read_csv(dataset_path)
    df['te'] = pd.to_datetime(df['te'])

    WINDOW_SIZE_MS = 10
    last_stamp = df.iloc[-1]['te']
    curr_stamp = df.iloc[0]['te']

    window_occurrences = {}

    i = 0
    while curr_stamp < last_stamp:
        gap = curr_stamp + pd.to_timedelta(WINDOW_SIZE_MS, unit='ms')
        occurrences = 0

        while i < len(df) and df.iloc[i]['te'] < gap:
            occurrences += 1
            i += 1
            if i < len(df):
                curr_stamp = df.iloc[i]['te']
            else:
                break

        if occurrences not in window_occurrences:
            window_occurrences[occurrences] = 0

        window_occurrences[occurrences] += 1

        print(f"curr_stamp: {curr_stamp}, last_stamp: {last_stamp}", end='\r')

    with open('saves/window_occurrences.pkl', 'rb') as file:
        window_occurrences = pickle.load(file)

    xs = sorted(window_occurrences.keys())[:100]
    ys = [window_occurrences[x] for x in xs]
    indices = list(range(len(xs)))


    plt.figure(figsize=(20, 10))
    plt.bar(indices, ys)
    plt.xlabel('Record Number')
    plt.ylabel('Occurrences')
    plt.title('Distribution of Occurrences for 10ms Window')
    plt.xticks(indices, xs, rotation=90)
    plt.tight_layout()

    plt.savefig('saves/fixed_window_10ms.png', bbox_inches='tight')
    plt.show()


def time_stamps_analisys():
    dataset_path = "dataset/UGR16/custom/train.csv"
    df = pd.read_csv(dataset_path)
    time_stamps = df['te'].value_counts()
    time_stamps = time_stamps.sort_index()

    xs = pd.to_datetime(time_stamps.index)
    ys = time_stamps.values


    plt.figure(figsize=(20, 10))
    plt.plot_date(xs, ys, linestyle='-')
    plt.xlabel('Time Stamp')
    plt.ylabel('Occurrences')
    plt.title('Occurrences for Time Stamps')
    plt.grid(True)
    plt.savefig('saves/time_stamps.png', bbox_inches='tight')
    plt.show()

    occurrences_count = time_stamps.value_counts().sort_index()
    print(occurrences_count)
    
    xs = occurrences_count.index.astype(int)
    ys = occurrences_count.values
    plt.figure(figsize=(20, 10))
    plt.bar(xs, ys)
    plt.xlabel('Sizes')
    plt.ylabel('Occurrences')
    plt.title('Sizes Occurrences')
    plt.tight_layout()

    plt.savefig('saves/occurrences_count.png', bbox_inches='tight')
    plt.show()

def main():
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )
    proportion_fix()


if __name__ == "__main__":
    main()