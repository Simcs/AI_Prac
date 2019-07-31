from lib.numerical_derivative import numerical_derivative
import numpy as np

class DeepLearning:
    def __init__(self, name, xdata, tdata, nodes, lr, epochs):
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

        self.W = []
        self.b = []
        for i in range(len(nodes) - 1):
            self.W.append(np.random.rand(nodes[i], nodes[i + 1]))
            self.b.append(np.random.rand(nodes[i + 1]))
        
        self.learning_rate = lr
        self.epochs = epochs

    def train(self, debug=False, interval=5):
        for step in range(self.epochs):
            for i in range(len(self.xdata)):
                self.input_data = self.xdata[i]
                self.target_data = self.tdata[i]

                f = lambda x : self.feed_forward()
                for i in range(len(self.W)):
                    self.W[i] -= self.learning_rate * numerical_derivative(f, self.W[i])
                    self.b[i] -= self.learning_rate * numerical_derivative(f, self.b[i])
            if debug and step % interval == 0:
                print("epoch:", step, "loss_val:", self.loss_val())
    
    def predict(self, test_data):
        z = np.dot(test_data, self.W[0]) + self.b[0]
        a = self.sigmoid(z)
        for i in range(1, len(self.W) - 1):
            z = np.dot(a, self.W[i]) + self.b[i]
            a = self.sigmoid(a)
        y = a
        print(y)
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
        z = np.dot(self.input_data, self.W[0]) + self.b[0]
        a = self.sigmoid(z)
        for i in range(len(self.W) - 1):
            z = np.dot(a, self.W[i + 1]) + self.b[i + 1]
            a = self.sigmoid(a) 
        y = a
        return (-1) * np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    def loss_val(self):
        delta = 1e-7
        z = np.dot(self.input_data, self.W[0]) + self.b[0]
        a = self.sigmoid(z)
        for i in range(len(self.W) - 1):
            z = np.dot(a, self.W[i + 1]) + self.b[i + 1]
            a = self.sigmoid(a) 
        y = a
        return (-1) * np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))
       
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))