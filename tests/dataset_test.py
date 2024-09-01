import pandas as pd

import torch


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

print(tensor[:, :2])
print(tensor[:, 2:])
