import numpy as np

def load_data(data_path):
    input = np.loadtxt(data_path)
    data = np.reshape(input,(1,96))

    return data

def load_label(label_path):
    input = np.loadtxt(label_path)
    data = data = np.reshape(input,(1,24))

    return data


