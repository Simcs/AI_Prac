from lib.numerical_derivative import numerical_derivative
from lib.data_generator import DataGenerator
import numpy as np

class DeepLearning:
    def __init__(self, name, i_node, h1_node, o_node, lr):
        self.name = name
        
        self.W2 = np.random.rand(i_node, h1_node)
        self.b2 = np.random.rand(h1_node)
        self.W3 = np.random.rand(h1_node, o_node)
        self.b3 = np.random.rand(o_node)

        self.learning_rate = lr

    def train(self, input, target):
        self.input_data = input
        self.target_data = target

        f = lambda x : self.feed_forward()
        self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
        self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
        self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
        self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)
    
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

    (training_data, test_data) = DataGenerator("ThoracicSurgery", "./data/ThoracicSurgery.csv", 0.6, True).generate()

    i_node = training_data.shape[1] - 1
    h1_node = 10
    o_node = 1
    lr = 1e-3
    epoch = 20

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

    