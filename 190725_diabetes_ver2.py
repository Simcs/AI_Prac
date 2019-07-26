from lib.numerical_derivative import numerical_derivative
from datetime import datetime
import numpy as np

class Diabetes:
    def __init__(self, name, i_node, h1_node, o_node, lr):
        self.name = name
        
        self.input_node = i_node
        self.h1_node = h1_node
        self.output_node = o_node

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
    diabetes_data = np.loadtxt("./data/diabetes.csv", delimiter=",", dtype=np.float32)
    train_rate = 0.6
    boundary = int(np.trunc(len(diabetes_data) * train_rate))

    train_data = diabetes_data[:boundary, :]
    test_xdata = diabetes_data[boundary:, 0:-1]
    test_tdata = diabetes_data[boundary:, -1]
    
    i_node = train_data.shape[1] - 1
    h1_node = 10
    o_node = 1
    lr = 1e-2
    epochs = 30

    test = Diabetes("Diabetes", i_node, h1_node, o_node, lr)
    for step in range(epochs):
        for i in range(len(train_data)):
            input_data = train_data[i, 0:-1]
            target_data = train_data[i, -1]
            test.train(input_data, target_data)
        if step % 5 == 0:
            print("epoch:", step, ", loss value:", test.loss_val())
        

    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))


    