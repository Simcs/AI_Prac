from lib.deep_learning import DeepLearning
from lib.data_generator import DataGenerator
import numpy as np

if __name__ == '__main__':

    (training_data, test_data) = DataGenerator("ThoracicSurgery", "./data/ThoracicSurgery.csv", 0.6, True).generate()

    i_node = training_data.shape[1] - 1
    h1_node = 10
    o_node = 1
    lr = 1e-3
    epochs = 20

    training_xdata = training_data[:, 0:-1]
    training_tdata = training_data[:, [-1]]
    test = DeepLearning("ThoracicSurgery", training_xdata, training_tdata, i_node, h1_node, o_node, lr, epochs)
    test.train(debug=True, interval=5)

    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]

    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))