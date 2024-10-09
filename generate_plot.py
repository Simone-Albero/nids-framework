from matplotlib import pyplot as plt
import pandas as pd
from io import StringIO
import os

def plot_1():
    data_1 = """Precision,Recall,F1,Balanced Accuracy,TN,FN,FP,TP
    0.9840674789128397,0.6375227686703097,0.7737656595431098,0.8169968005693256,4800,17,597,1050
    0.9815261044176706,0.7419550698239222,0.8450899031811895,0.8685901568758391,4794,23,425,1222
    0.972733469665985,0.8664238008500303,0.9165061014771997,0.9290599386230638,4777,40,220,1427
    0.8566176470588235,0.9902853673345476,0.9186144747958321,0.9668055443689554,4544,273,16,1631
    0.9425556858147714,0.97632058287796,0.9591410677005666,0.977987985024199,4719,98,39,1608
    0.9457364341085271,0.9629629629629629,0.9542719614921781,0.972035768382042,4726,91,61,1586
    0.9408695652173913,0.9854280510018215,0.9626334519572953,0.9821265229059346,4715,102,24,1623
    0.9436288901937757,0.9757134183363692,0.959402985074627,0.9778920008435011,4721,96,40,1607
    0.9456585942114589,0.9720704310868246,0.9586826347305389,0.976485703398924,4725,92,46,1601
    0.9422740524781341,0.9811778992106861,0.9613325401546698,0.9803128441455133,4718,99,31,1616
    0.9363166953528399,0.9908925318761385,0.9628318584070796,0.983924571937654,4706,111,15,1632
    """

    data_2 = """Precision,Recall,F1,Balanced Accuracy,TN,FN,FP,TP
    0.9134328358208955,0.18579234972677597,0.3087790110998991,0.5898860025569732,4788,29,1341,306
    0.5990230905861457,0.8190649666059502,0.6919723005898948,0.8158019456239217,3914,903,298,1349
    0.617658498638662,0.9641772920461446,0.752963489805595,0.8800541847401161,3834,983,59,1588
    0.9484151646985706,0.9265330904675168,0.9373464373464374,0.9546512244947092,4734,83,121,1526
    0.8548812664907651,0.9836065573770492,0.9147374364765669,0.9632585413001086,4542,275,27,1620
    0.7841346153846154,0.9902853673345476,0.8752347732760933,0.9485369124403691,4368,449,16,1631
    0.9404553415061296,0.9781420765027322,0.9589285714285714,0.9784835356563899,4715,102,36,1611
    0.8949972512369434,0.9884638737097754,0.9394114252740912,0.9744063192505696,4626,191,19,1628
    0.9487790351399643,0.9672131147540983,0.957907396271798,0.9746798395028535,4731,86,54,1593
    0.8625592417061612,0.994535519125683,0.9238578680203046,0.9701762088051085,4556,261,9,1638
    0.9300578034682081,0.9769277474195507,0.9529167900503406,0.9759041892588722,4696,121,38,1609
    """

    df_1 = pd.read_csv(StringIO(data_1))
    df_2 = pd.read_csv(StringIO(data_2))
    x_ticks = range(40, 150, 10)

    plt.figure(figsize=(12, 8))

    colors_self_supervised = {
        'Precision': '#1f77b4',
        'Recall': '#2ca02c',
        'F1': 'red',
        'Balanced Accuracy': 'orange'
    }

    colors_supervised = {
        'Precision': 'DodgerBlue',
        'Recall': 'green',
        'F1': 'red',
        'Balanced Accuracy': 'orange'
    }

    #metrics = ['Precision', 'Recall']
    metrics = ['F1', 'Balanced Accuracy']

    # (Self-Supervised)
    for metric in metrics:
        plt.plot(x_ticks, df_1[metric], marker='o', linestyle='-', color=colors_self_supervised[metric], label=f'{metric} (Self-Supervised)')

    # (Supervised)
    for metric in metrics:
        plt.plot(x_ticks, df_2[metric], marker='x', linestyle='--', color=colors_supervised[metric], label=f'{metric} (Supervised)')

    plt.xlabel('Batches', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.xticks(x_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, title='Metrics', title_fontsize='16')
    plt.grid(True, linestyle='--') 

    plot_filename = os.path.join("figures", "plot_2.png")
    plt.savefig(plot_filename)
    plt.close()

def plot_2():
    data = """Precision,Recall,F1,Balanced Accuracy,TN,FN,FP,TP
    0.9363166953528399,0.9908925318761385,0.9628318584070796,0.983924571937654,4706,111,15,1632
    0.9564950980392157,1.0,0.9777638584403383,0.9924644449161537,4640,71,0,1561
    0.9506172839506173,1.0,0.9746835443037974,0.9910998092816274,4635,84,0,1617
    """

    df = pd.read_csv(StringIO(data))
    #columns = ['Precision', 'Recall', 'F1', 'Balanced Accuracy']
    columns = ['TN', 'FN', 'FP', 'TP']

    df_transposed = df[columns].T
    df_transposed.columns = ['time-based', 'ip-src-based', 'ip-dst-based']

    df_transposed.plot(kind='bar', figsize=(10, 6))

    plt.xlabel('Metric', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(title='Context Window', fontsize=16, loc='lower left', title_fontsize='16')
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.grid(True, linestyle='--') 

    plot_filename = os.path.join("figures", "plot_2.png")
    plt.savefig(plot_filename)
    plt.close()


def plot_3():
    data_1 = """Class,Precision,Recall,F1,Support
    Weighted,0.7865,0.7943,0.7807,6912
    Weighted,0.8189,0.8326,0.8158,6912
    Weighted,0.8271,0.8378,0.8231,6912
    Weighted,0.8317,0.8626,0.8408,6912
    Weighted,0.8351,0.8440,0.8295,6912
    Weighted,0.8257,0.8377,0.8223,6912
    Weighted,0.8234,0.8514,0.8313,6912
    Weighted,0.8295,0.8422,0.8263,6912
    """

    data_2 = """Class,Precision,Recall,F1,Support
    Weighted,0.6171,0.5245,0.5222,6912
    Weighted,0.7727,0.8019,0.7808,6912
    Weighted,0.8210,0.7865,0.7767,6912
    Weighted,0.8136,0.8278,0.8125,6912
    Weighted,0.8294,0.8372,0.8228,6912
    Weighted,0.8299,0.8359,0.8225,6912
    Weighted,0.8196,0.8247,0.8120,6912
    Weighted,0.8275,0.8336,0.8204,6912
    """

    df_1 = pd.read_csv(StringIO(data_1))
    df_2 = pd.read_csv(StringIO(data_2))
    x_ticks = range(100, 500, 50)

    plt.figure(figsize=(12, 8))

    colors_self_supervised = {
        'Precision': '#1f77b4',
        'Recall': '#2ca02c',
        'F1': 'red',
    }

    colors_supervised = {
        'Precision': '#1f77b4',
        'Recall': '#2ca02c',
        'F1': 'red',
    }

    #metrics = ['Precision', 'Recall']
    metrics = ['F1']

    # (Self-Supervised)
    for metric in metrics:
        plt.plot(x_ticks, df_1[metric], marker='o', linestyle='-', color=colors_self_supervised[metric], label=f'{metric} (Self-Supervised)')

    # (Supervised)
    for metric in metrics:
        plt.plot(x_ticks, df_2[metric], marker='x', linestyle='--', color=colors_supervised[metric], label=f'{metric} (Supervised)')

    plt.xlabel('Batches', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.xticks(x_ticks, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, title='Metrics', title_fontsize='16')
    plt.grid(True, linestyle='--') 

    plot_filename = os.path.join("figures", "plot_3.png")
    plt.savefig(plot_filename)
    plt.close()

def plot_4():
    data = """Class,Precision,Recall,F1,Support
    Weighted,0.8295,0.8422,0.8263,6912
    Weighted,0.8652,0.8692,0.8551,6720
    Weighted,0.8865,0.8821,0.8696,6656
    """

    df = pd.read_csv(StringIO(data))
    columns = ['Precision', 'Recall', 'F1']

    df_transposed = df[columns].T
    df_transposed.columns = ['time-based', 'ip-src-based', 'ip-dst-based']

    df_transposed.plot(kind='bar', figsize=(10, 6))

    plt.xlabel('Metric', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.legend(title='Context Window', fontsize=16, loc='lower left', title_fontsize='16')
    plt.xticks(rotation=0, fontsize=16)
    plt.yticks(rotation=0, fontsize=16)
    plt.grid(True, linestyle='--') 

    plot_filename = os.path.join("figures", "plot_4.png")
    plt.savefig(plot_filename)
    plt.close()

def plot_5():
    data_1 = """Class,Precision,Recall,F1,Support
    0,0.9931,0.9647,0.9787,4017
    1,0.4323,0.9391,0.5921,493
    2,0.9480,0.7675,0.8483,499
    3,0.7085,0.8545,0.7747,495
    4,0.5445,0.7126,0.6173,421
    5,0.0000,0.0000,0.0000,496
    6,0.9742,0.7678,0.8588,491
    """

    data_2 = """Class,Precision,Recall,F1,Support
    0,1.0000,0.9754,0.9876,3868
    1,0.5104,0.9383,0.6612,470
    2,0.9511,0.7907,0.8635,492
    3,0.8287,0.9518,0.8860,498
    4,0.5543,0.7867,0.6503,422
    5,0.2410,0.0412,0.0703,486
    6,0.9810,0.8512,0.9115,484
    """

    data_3 = """Class,Precision,Recall,F1,Support
    0,1.0000,0.9747,0.9872,3877
    1,0.5645,0.8240,0.6700,409
    2,0.9493,0.8004,0.8685,491
    3,0.8384,0.9920,0.9088,497
    4,0.5564,0.8690,0.6784,420
    5,0.5455,0.1222,0.1997,491
    6,0.8672,0.9427,0.9034,471
    """

    metric = 'Recall'
    df_1 = pd.read_csv(StringIO(data_1))[['Class', metric]]
    df_2 = pd.read_csv(StringIO(data_2))[['Class', metric]]
    df_3 = pd.read_csv(StringIO(data_3))[['Class', metric]]

    df_merged = pd.DataFrame({
        'Class': df_1['Class'],
        'time-based': df_1[metric],
        'ip-src-based': df_2[metric],
        'ip-dst-based': df_3[metric]
    })

    class_names = ['Benign', 'Exploits', 'Generic', 'Fuzzers', 'Backdoor', 'DoS', 'Reconnaissance']
    df_merged['Class'] = class_names

    # Crea il grafico a barre raggruppato
    df_merged.plot(
        x='Class', 
        kind='bar', 
        figsize=(10, 6),
    )

    plt.xlabel('Class', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.xticks(rotation=20, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(['time-based', 'ip-src-based', 'ip-dst-based'], fontsize=16, loc='lower left', title='Context Window', title_fontsize='16')
    plt.grid(True, linestyle='--') 
    plt.tight_layout()

    plot_filename = os.path.join("figures", "plot_5.png")
    plt.savefig(plot_filename)
    plt.close()




if __name__ == "__main__":
    plot_5()
