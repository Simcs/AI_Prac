from lib.numerical_derivative import numerical_derivative
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

        self.W2 = np.random.rand(i_node, h1_node)
        self.b2 = np.random.rand(h1_node)
        self.W3 = np.random.rand(h1_node, o_node)
        self.b3 = np.random.rand(o_node)

        self.learning_rate = lr
        self.epochs = epochs

    def train(self, debug=False, interval=5):
        for step in range(self.epochs):
            for i in range(len(self.xdata)):
                self.input_data = self.xdata[i]
                self.target_data = self.tdata[i]

                f = lambda x : self.feed_forward()
                self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
                self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
                self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
                self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)
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