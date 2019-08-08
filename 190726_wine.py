from lib.deep_learning import DeepLearning
from lib.data_generator import DataGenerator
import numpy as np

if __name__ == '__main__':
    (training_data, test_data) = DataGenerator("wine", "./data/wine.csv", 0.7, True).generate()

    training_xdata = training_data[:, 0:-1]
    training_tdata = training_data[:, [-1]]
    
    i_node = training_xdata.shape[1]
    h1_node = 4
    o_node = 1
    lr = 1e-2
    epochs = 2

    test = DeepLearning("wine", training_xdata, training_tdata, i_node, h1_node, o_node, lr, epochs)
    test.train(debug=True, interval=5)

    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]

    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))
    