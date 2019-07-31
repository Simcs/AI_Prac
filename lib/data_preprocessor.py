import numpy as np
import os

def separate(file_name, rate, delimiter=","):
    data = np.loadtxt(file_name, delimiter=delimiter, dtype=np.float32)
    np.random.shuffle(data)

    boundary = int(np.trunc(len(data) * rate))
    training_data = data[:boundary, :]
    test_data = data[boundary:, :]
    
    path = os.path.splitext(file_name)[0]
    training_data_file_name = path + "_training_data.csv"
    test_data_file_name = path + "_test_data.csv"

def normalize(file_name, delimiter=","):

    data = np.loadtxt(file_name, delimiter=delimiter, dtype=np.float32)
    path = os.path.splitext(file_name)[0]
    new_file_name = path + "_normalized.csv"

    maximum = np.amax(data, axis=0)
    data = np.divide(data, maximum)

    np.savetxt(new_file_name, data, delimiter=delimiter)