import numpy as np
import os

class DataGenerator:
    def __init__(self, name, file_path, separation_rate, is_normalized=False):
        self.name = name
        self.file_path = file_path
        self.separation_rate = separation_rate
        self.is_normalized = is_normalized
    
    def normalize_data_using_max(self, load_data):
        data_max = np.amax(load_data, axis=0)
        # divide data by max value only if max value is bigger than 1.0
        for i in range(len(data_max)):
            data_max[i] = data_max[i] if data_max[i] > 1.0 else 1.0
        normalized_data = np.divide(load_data, data_max)
        return normalized_data
    
    def generate(self):
        load_data = np.loadtxt(self.file_path, delimiter=",", dtype=np.float32)
        if self.is_normalized:
            load_data = self.normalize_data_using_max(load_data)

        # shuffle data to choose random data : Fisher-Yates algotirhm
        np.random.shuffle(load_data)
        boundary = int(np.trunc(len(load_data) * self.separation_rate))
        training_data = load_data[:boundary, :]
        test_data = load_data[boundary:, :]

        path_without_ext = os.path.splitext(self.file_path)[0]
        if self.is_normalized:
            path_without_ext += "_normalized"
        training_data_file_path = path_without_ext + "_training_data.csv"
        test_data_file_path = path_without_ext + "_test_data.csv"

        np.savetxt(training_data_file_path, training_data, delimiter=",")
        np.savetxt(test_data_file_path, test_data, delimiter=",")

        return (training_data, test_data)


    