from lib.numerical_derivative import numerical_derivative
from lib.data_generator import DataGenerator
import numpy as np

class DeepLearning:
    def __init__(self, name, xdata, tdata, i_node, h1_node, o_node, lr, epochs):
        self.name = name
        if xdata.ndim == 1:     # vector
            self.xdata = xdata.reshape(len(xdata), 1)
            self.tdata = tdata.reshape(len(tdata), 1)
        else:
            self.xdata = xdata
            if tdata.ndim == 1: # vector
                self.tdata = tdata.reshape(len(tdata), 1) 
            else:
                self.tdata = tdata

        self.input_node = i_node
        self.h1_node = h1_node
        self.output_node = o_node

        # Xavier/He initialization
        self.W2 = np.random.randn(i_node, h1_node) / np.sqrt(self.input_node / 2)
        self.b2 = np.random.rand(1, h1_node)
        self.W3 = np.random.randn(h1_node, o_node) / np.sqrt(self.h1_node / 2)
        self.b3 = np.random.rand(1, o_node)

        self.learning_rate = lr
        self.epochs = epochs

    def train(self, debug=False, interval=5):
        for step in range(self.epochs):
            for i in range(len(self.xdata)):
                self.input_data = np.array(self.xdata[i], ndmin=2)
                self.target_data = np.array(self.tdata[i], ndmin=2)

                # print("input", self.input_data.shape)
                z2 = np.dot(self.input_data, self.W2) + self.b2
                # print("z2", z2.shape)
                a2 = self.sigmoid(z2)
                # print("a2", a2.shape)
                z3 = np.dot(a2, self.W3) + self.b3
                # print("z3", z3.shape)
                a3 = self.sigmoid(z3)
                # print("a3", a3.shape)

                loss_3 = (a3 - self.target_data) * a3 * (1 - a3)
                self.W3 -= self.learning_rate * np.dot(a2.T, loss_3)
                self.b3 -= self.learning_rate * loss_3
                # print("loss_3", loss_3.shape)
                loss_2 = np.dot(loss_3, self.W3.T) * a2 * (1 - a2)
                self.W2 -= self.learning_rate * np.dot(self.input_data.T, loss_2)
                self.b2 -= self.learning_rate * loss_2
                # print("loss_2", loss_2.shape)

            if debug and step % interval == 0:
                print("epoch:", step, "loss_val:", self.loss_val())
    
    def predict(self, test_data):
        z2 = np.dot(test_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result
    
    def accuracy(self, xdata, tdata):
        matched_list = []
        not_matched_list = []
        index_label_prediction_list = []
        
        result = [self.predict(data)[1] for data in xdata]
        for i in range(len(result)):
            if result[i] == tdata[i]:
                matched_list.append(i)
            else:
                not_matched_list.append(i)
            index_label_prediction_list.append([i, tdata[i], result[i]])

        return matched_list, not_matched_list, index_label_prediction_list

    def feed_forward(self):
        delta = 1e-7
        z2 = np.dot(self.input_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    def loss_val(self):
        delta = 1e-7
        z2 = np.dot(self.input_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))
       
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == '__main__':

    # (training_data, test_data) = DataGenerator("Diabetes", "./data/diabetes.csv", 0.5, True).generate()
    # training_xdata = training_data[:, 0:-1]
    # training_tdata = training_data[:, -1]
    # test_xdata = test_data[:, 0:-1]
    # test_tdata = test_data[:, -1]
    
    # test = DeepLearning("Diabetes", training_xdata, training_tdata, i_node, h1_node, o_node, lr, epochs)
    # test.train(debug=True, interval=50)

    # (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    # print(prediction_list)
    # print("accuracy:", len(matched_list) / len(test_xdata))

    (training_data, test_data) = DataGenerator("ThoracicSurgery", "./data/ThoracicSurgery.csv", 0.6, True).generate()
    training_xdata = training_data[:, 0:-1]
    training_tdata = training_data[:, -1]
    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]

    i_node = training_xdata.shape[1]
    h1_node = 50
    o_node = 1
    lr = 1e-1
    epochs = 500

    test = DeepLearning("ThoracicSurgery", training_xdata, training_tdata, i_node, h1_node, o_node, lr, epochs)
    test.train(debug=True, interval=50)

    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))