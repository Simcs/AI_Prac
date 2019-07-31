import numpy as np
from lib.deep_learning_ver2 import DeepLearning
import lib.data_preprocessor

if __name__ == '__main__':
    lib.data_preprocessor.normalize("./data/wine.csv")
    lib.data_preprocessor.separate("./data/wine_normalized.csv", 0.4)

    training_data = np.loadtxt("./data/wine_normalized_training_data.csv", delimiter=",", dtype=np.float32)
    test_data = np.loadtxt("./data/wine_normalized_test_data.csv", delimiter=",", dtype=np.float32)

    training_xdata = training_data[:, 0:-1]
    training_tdata = training_data[:, -1]
    i_node = training_xdata.shape[1]
    h1_node = 1
    o_node = 1
    lr = 1e-2
    epochs = 2
    test = DeepLearning("wine", training_xdata, training_tdata, [i_node, h1_node, o_node], lr, epochs)
    test.train(True, 5)

    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]
    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))
    