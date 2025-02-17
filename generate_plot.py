from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os

def plot_E1(hybrid_data_path, baseline_data_path, file_name):

    df_hybrid = pd.read_csv(hybrid_data_path)
    df_baseline = pd.read_csv(baseline_data_path)

    x_ticks = list(map(lambda x: f"{int(x*32*10/1000) if (x*32*10/1000) % 1 == 0 else x*32*10/1000}", range(10, 310, 10))) 
    # x_ticks = range(10, 310, 10)


    colors = {
        'Hybrid': {
            'Precision': '#C8CFA0',
            'Recall': '#78ABA8',
            'F1': '#EF9C66',
            'Balanced Accuracy': '#FCDC94'
        },
        'Baseline': {
            'Precision': '#78ABA8',
            'Recall': '#C8CFA0',
            'F1': '#78ABA8',
            'Balanced Accuracy': '#EF9C66'
        }
    }

    metrics = ['F1']

    plt.figure(figsize=(16, 8))

    styles = {
        'Hybrid': {'linestyle': '-', 'marker': 'o'},
        'Baseline': {'linestyle': '--', 'marker': '^'}
    }

    for approach, df in [('Hybrid', df_hybrid), ('Baseline', df_baseline)]:
        for metric in metrics:
            plt.plot(
                x_ticks, df[metric],
                linestyle=styles[approach]['linestyle'],
                marker=styles[approach]['marker'],
                color=colors[approach][metric],
                label=approach
            )

    plt.xlabel('Thousands of Labeled Samples', fontsize=20)
    plt.xticks(x_ticks, fontsize=16, rotation=90)

    plt.rc('font', size=14)
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    y_min, y_max = 0, 1
    y_interval = 0.1
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(y_min, y_max + y_interval, y_interval), fontsize=16)
    # plt.yticks(fontsize=16)
    plt.ylabel('F1 Score', fontsize=20)

    plt.legend(fontsize=20, loc="lower right")
    plt.grid(True, linestyle='--')

    plot_filename = os.path.join("figures", file_name)
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()


def plot_E2(NB15_path, ToN_path, CIC_path, file_name):
    
    NROWS=25
    df_NB15 = pd.read_csv(NB15_path, nrows=NROWS)
    df_ToN = pd.read_csv(ToN_path, nrows=NROWS)
    df_CIC = pd.read_csv(CIC_path, nrows=NROWS)

    x_ticks = list(map(lambda x: f"{int(x*32*10/1000)}", range(100, 100*NROWS+100, 100))) 
    #x_ticks = range(100, 3100, 100)
    

    colors = {
        "NB15":{
            'Precision': '#C8CFA0',
            'Recall': '#78ABA8',
            'F1': '#EF9C66',
            'Balanced Accuracy': '#FCDC94'
        },

        "ToN":{
            'Precision': '#C8CFA0',
            'Recall': '#EF9C66',
            'F1': '#FA5053',
            'Balanced Accuracy': '#FCDC94'
        },

        "CIC":{
        'Precision': '#EF9C66',
        'Recall': '#78ABA8',
        'F1': '#C8CFA0',
        'Balanced Accuracy': '#FCDC94'
        }
    }

    metrics = ['F1']

    plt.figure(figsize=(16, 8))

    styles = {
        "NB15":{'linestyle': '-', 'marker': 'o'},
        "ToN":{'linestyle': '-', 'marker': 's'},
        "CIC":{'linestyle': '-', 'marker': '^'},

    }

    for metric in metrics:
        plt.plot(
            x_ticks, df_NB15[metric],
            linestyle=styles["NB15"]['linestyle'],
            marker=styles["NB15"]['marker'],
            color=colors["NB15"][metric],
            label="NB15"
        )

    for metric in metrics:
        plt.plot(
            x_ticks, df_ToN[metric],
            linestyle=styles["ToN"]['linestyle'],
            marker=styles["ToN"]['marker'],
            color=colors["ToN"][metric],
            label="ToN"
        )

    for metric in metrics:
        plt.plot(
            x_ticks, df_CIC[metric],
            linestyle=styles["CIC"]['linestyle'],
            marker=styles["CIC"]['marker'],
            color=colors["CIC"][metric],
            label="CIC"
        )

    plt.xlabel('Thousands of Unlabeled Samples', fontsize=20)
    plt.xticks(x_ticks, fontsize=16, rotation=90)

    plt.rc('font', size=8)
    plt.rcParams['hatch.linewidth'] = 0.3
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    y_min, y_max = 0, 1
    y_interval = 0.1
    plt.ylim(y_min, y_max)
    plt.yticks(np.arange(y_min, y_max + y_interval, y_interval), fontsize=16)
    # plt.yticks(fontsize=16)
    plt.ylabel('F1 Score', fontsize=20)

    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--')

    plot_filename = os.path.join("figures", file_name)
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    NB15_BASELINE = "logs/NB15/baseline_small_NB15_post_10B.csv"
    NB15_HYBRID = "logs/NB15/hybrid_small_NB15_post_10B.csv"
    ToN_BASELINE = "logs/ToN/baseline_small_ToN_post_10B.csv"
    ToN_HYBRID = "logs/ToN/hybrid_small_ToN_post_10B.csv"
    CIC_BASELINE = "logs/CIC/baseline_small_CIC_post_10B.csv"
    CIC_HYBRID = "logs/CIC/hybrid_small_CIC_post_10B.csv"

    NB15_PT = "logs/NB15/pretraining_25b_small_NB15.csv"
    ToN_PT = "logs/ToN/pretraining_25b_small_ToN.csv"
    CIC_PT = "logs/CIC/pretraining_25b_small_CIC.csv"


    plot_E1(NB15_HYBRID, NB15_BASELINE, "NB15_10B.pdf")
    plot_E1(ToN_HYBRID, ToN_BASELINE, "ToN_10B.pdf")
    plot_E1(CIC_HYBRID, CIC_BASELINE, "CIC_10B.pdf")

    plot_E2(NB15_PT, ToN_PT, CIC_PT, "pretraining_25b.pdf")
