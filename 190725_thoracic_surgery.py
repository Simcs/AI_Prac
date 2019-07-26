from lib.deep_learning import DeepLearning
import lib.data_preprocessor
import numpy as np

if __name__ == '__main__':
    lib.data_preprocessor.normalize("./data/ThoracicSurgery.csv")
    lib.data_preprocessor.separate("./data/ThoracicSurgery_normalized.csv", 0.6)

    training_data = np.loadtxt("./data/ThoracicSurgery_normalized_training_data.csv", delimiter=",", dtype=np.float32)
    test_data = np.loadtxt("./data/ThoracicSurgery_normalized_test_data.csv", delimiter=",", dtype=np.float32)
    
    i_node = training_data.shape[1] - 1
    h1_node = 10
    o_node = 1
    lr = 1e-2
    epoch = 10

    test = DeepLearning("ThoracicSurgery", i_node, h1_node, o_node, lr)
    for i in range(epoch):
        for j in range(len(training_data)):
            input_data = training_data[i, 0:-1]
            target_data = training_data[i, -1]
            test.train(input_data, target_data)
        if i % 5 == 0:
            print("epoch:", i, ", loss value:", test.loss_val())

    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]

    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))

    