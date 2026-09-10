import pandas as pd
import matplotlib.pyplot as plt

import os

data_path = "../data"


if __name__ == "__main__":

    data_files = os.listdir(data_path);

    datas = []
    for path in [os.path.join(data_path, data_file) for data_file in data_files]:
        data = pd.read_csv(path, delimiter=";", index_col='size')
        # data = data.set_index(data['size'])
        col = path.split('/')[-1]

        data[col] = data['avg']
        data = data[col]
        datas.append(data)
        print(data)
        # break

    data = pd.concat(datas, axis=1)
    data.plot()

    plt.xlabel("Matrix Size (N x N)")
    plt.ylabel("Time (ms)")
    plt.title("Algorithms Comparison")
    plt.grid(True)
    plt.savefig("./plot.png")
        
