import logging
import os

import pandas as pd
from rich.logging import RichHandler
import torch
import numpy as np
import matplotlib.pyplot as plt

def indices_test():
    data = {
        'A': [28, 34, 22, 45, 30],
        'B': [2500, 4000, 1500, 5500, 3000]
    }

    df = pd.DataFrame(data)

    tensor = torch.tensor(df.values)


    index = df.index.tolist()
    print(index)

    print(df.iloc[index[2]].tolist(), tensor[index[2]].tolist())
    print(df.iloc[index[2]].tolist() == tensor[index[2]].tolist())


    tensor = torch.tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16],
                        [17, 18, 19, 20]])

def temporal_analysis():
    file_path = "../datasets/NF-ToN-IoT-V2/NF-ToN-IoT-v2.csv"
    df = pd.read_csv(file_path)
    
    target_column = "Attack"
    unique_labels = df[target_column].unique()
    logging.info(f"Unique Labels in '{target_column}' column: {unique_labels}")

    logging.info("Stats based on index distances between consecutive occurrences:")
    for label in unique_labels:
        label_indices = df[df[target_column] == label].index
        
        distances = label_indices.to_series().diff().dropna()

        distance_counts = distances.value_counts()
        if len(distance_counts) > 100:
                distance_counts = distance_counts.head(100)
        mean_distance = np.average(distance_counts.index, weights=distance_counts.values)
            
        distance_counts = distance_counts.sort_index()
        logging.info(f"Label '{label}' - Mean Distance: {mean_distance}")

        plt.figure(figsize=(12, 8))
        plt.bar(distance_counts.index, distance_counts.values, color='skyblue', width=0.4, align='center')
        
        plt.axvline(mean_distance, color='red', linestyle='--', label=f'Mean Distance: {mean_distance:.2f}')
        
        plt.xscale('log')
        plt.yscale('log')

        plt.grid(True, which="both", linestyle='--', linewidth=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.xlabel('Distance Between Occurrences')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Distances for Label: {label}')
        plt.legend()
        plt.tight_layout()
        
        plot_filename = os.path.join("plots", f'{label}_temporal_distances.png')
        plt.savefig(plot_filename)
        logging.info(f"Plot saved as {plot_filename}")
        plt.close()

def spatial_analysis():
    file_path = "../datasets/NF-ToN-IoT-V2/NF-ToN-IoT-v2.csv"
    df = pd.read_csv(file_path)
    
    target_column = "Attack"
    unique_labels = df[target_column].unique()
    logging.info(f"Unique Labels in '{target_column}' column: {unique_labels}")

    logging.info("Stats based on index distances between consecutive occurrences:")
    for label in unique_labels:
        label_df = df[df[target_column] == label]
        feature = "L4_DST_PORT"
        feature_uniques = label_df[feature].value_counts().index[:50]

        distances = pd.Series(dtype=float)

        for unique in feature_uniques:
            unique_df = label_df[label_df[feature] == unique]
            ip_distances = unique_df.index.to_series().diff().dropna()
            distances = pd.concat([distances, ip_distances], ignore_index=True)
        

        distance_counts = distances.value_counts()
        if len(distance_counts) > 100:
                distance_counts = distance_counts.head(100)
        mean_distance = np.average(distance_counts.index, weights=distance_counts.values)
            
        distance_counts = distance_counts.sort_index()
        logging.info(f"Label '{label}' - Mean Distance: {mean_distance}")

        plt.figure(figsize=(12, 8))
        plt.bar(distance_counts.index, distance_counts.values, color='skyblue', width=0.4, align='center')
        
        plt.axvline(mean_distance, color='red', linestyle='--', label=f'Mean Distance: {mean_distance:.2f}')
        
        plt.xscale('log')
        plt.yscale('log')

        plt.grid(True, which="both", linestyle='--', linewidth=0.7)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.xlabel('Distance Between Occurrences')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Distances for {feature}: {label}')
        plt.legend()
        plt.tight_layout()
        
        plot_filename = os.path.join("plots", f'{label}_{feature}_distances.png')
        plt.savefig(plot_filename)
        logging.info(f"Plot saved as {plot_filename}")
        plt.close()


if __name__ == "__main__":
    debug_level = logging.INFO
    logging.basicConfig(
        level=debug_level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)],
    )

    spatial_analysis()