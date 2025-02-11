from matplotlib import pyplot as plt
import pandas as pd
from io import StringIO
import os

def plot():
    data_hybrid = """Precision,Recall,F1,Balanced Accuracy,TN,FN,FP,TP
0.814,0.877,0.844,0.931,436206,6973,4286,30551
0.756,0.956,0.844,0.966,432444,10735,1548,33289
0.650,0.971,0.779,0.965,424962,18217,1005,33832
0.770,0.975,0.860,0.976,433027,10152,864,33973
0.867,0.949,0.906,0.969,438125,5054,1777,33060
0.697,0.991,0.818,0.979,428163,15016,314,34523
0.646,0.994,0.783,0.975,424203,18976,220,34617
0.828,0.993,0.903,0.988,435993,7186,249,34588
0.863,0.979,0.917,0.983,437749,5430,731,34106
0.587,0.997,0.739,0.971,418743,24436,102,34735
0.698,0.995,0.820,0.980,428205,14974,187,34650
0.881,0.980,0.928,0.985,438580,4599,697,34140
0.861,0.981,0.917,0.984,437665,5514,656,34181
0.825,0.993,0.902,0.988,435845,7334,227,34610
0.854,0.982,0.913,0.984,437327,5852,640,34197
0.854,0.981,0.913,0.984,437347,5832,648,34189
0.795,0.988,0.881,0.984,434319,8860,420,34417
0.860,0.983,0.917,0.985,437582,5597,592,34245
0.808,0.998,0.893,0.990,434937,8242,80,34757
0.859,0.982,0.917,0.985,437554,5625,611,34226
    """

    data_baseline = """Precision,Recall,F1,Balanced Accuracy,TN,FN,FP,TP
0.967,0.039,0.074,0.519,443133,46,33492,1345
0.891,0.758,0.819,0.876,439944,3235,8416,26421
0.852,0.947,0.897,0.967,437443,5736,1850,32987
0.760,0.936,0.839,0.957,432871,10308,2219,32618
0.858,0.952,0.902,0.970,437678,5501,1677,33160
0.859,0.960,0.906,0.974,437672,5507,1400,33437
0.765,0.967,0.855,0.972,432841,10338,1138,33699
0.764,0.969,0.855,0.973,432760,10419,1075,33762
0.833,0.972,0.897,0.978,436396,6783,990,33847
0.751,0.977,0.849,0.976,431909,11270,801,34036
0.817,0.982,0.892,0.982,435530,7649,634,34203
0.833,0.974,0.898,0.979,436393,6786,922,33915
0.833,0.978,0.900,0.981,436369,6810,775,34062
0.830,0.979,0.898,0.982,436192,6987,738,34099
0.840,0.969,0.900,0.977,436726,6453,1071,33766
0.839,0.973,0.901,0.979,436664,6515,957,33880
0.824,0.986,0.898,0.985,435838,7341,492,34345
0.820,0.980,0.893,0.982,435707,7472,688,34149
0.818,0.981,0.892,0.982,435603,7576,676,34161
0.823,0.981,0.895,0.982,435819,7360,673,34164
    """

    df_hybrid = pd.read_csv(StringIO(data_hybrid))
    df_baseline = pd.read_csv(StringIO(data_baseline))
    x_ticks = range(50, 1001, 50)

    plt.figure(figsize=(12, 8))

    hybrid_colors = {
        'Precision': '#C8CFA0',
        'Recall': '#78ABA8',
        'F1': '#EF9C66',
        'Balanced Accuracy': '#FCDC94'
    }


    baseline_colors = {
        'Precision': '#78ABA8',
        'Recall': '#C8CFA0',
        'F1': '#78ABA8',
        'Balanced Accuracy': '#EF9C66'
    }

    #metrics = ['Precision', 'Recall']
    #metrics = ['F1', 'Balanced Accuracy']
    metrics = ['F1']
    # (hybrid)
    for metric in metrics:
        plt.plot(x_ticks, df_hybrid[metric], marker='o', linestyle='-', color=hybrid_colors[metric], label=f'hybrid')

    # (Baseline)
    for metric in metrics:
        plt.plot(x_ticks, df_baseline[metric], marker='x', linestyle='-', color=baseline_colors[metric], label=f'supervised')

    plt.xlabel('Batches', fontsize=20)
    plt.ylabel('F1', fontsize=20)
    plt.xticks(x_ticks, fontsize=16, rotation=30)
    plt.yticks(fontsize=16)
    plt.ylim([0, 1])
    plt.legend(fontsize=16, title='Approach', title_fontsize='16')
    plt.grid(True, linestyle='--') 

    plot_filename = os.path.join("figures", "NB15_medium.png")
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    plot()
