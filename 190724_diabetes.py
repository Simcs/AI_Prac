from lib.numerical_derivative import numerical_derivative
from lib.data_generator import DataGenerator
from datetime import datetime
import numpy as np

class Diabetes:
    def __init__(self, name, xdata, tdata, i_node, h1_node, o_node, lr, itr_count):
        self.name = name
        if xdata.ndim == 1:
            self.xdata = xdata.reshape(len(xdata), 1)
            self.tdata = tdata.reshape(len(tdata), 1)
        elif xdata.ndim == 2:
            self.xdata = xdata
            if tdata.ndim == 1:
                self.tdata = tdata.reshape(len(tdata), 1)
            else:
                self.tdata = tdata
        
        self.W2 = np.random.rand(i_node, h1_node)
        self.b2 = np.random.rand(h1_node)
        self.W3 = np.random.rand(h1_node, o_node)
        self.b3 = np.random.rand(o_node)

        self.learning_rate = lr
        self.iteration_count = itr_count

    def train(self, debug=False, interval=1000):
        f = lambda x : self.feed_forward()
        start_time = datetime.now()
        
        if debug:
            print("Test System :")
            print("learning rate:", self.learning_rate, "iteration count:", self.iteration_count)
            print(self.name, "initial loss value :", self.loss_val(), "\n")

        for step in range(self.iteration_count + 1):
            self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
            self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
            self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
            self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)
            if debug and step % interval == 0:
                print(self.name, "step :", step, ", loss_value :", self.loss_val())
        
        if debug:
            print(self.name, 'train result:')
            print("final loss value :", self.loss_val())
            print("final W2 =", self.W2)
            print("final b2 =", self.b2)
            print("final W3 =", self.W3)
            print("final b3 =", self.b3)
            print(self.name, "Elapsed time :", datetime.now() - start_time, "\n")
    
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
        z2 = np.dot(self.xdata, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def feed_forward2(self, x, t):
        delta = 1e-7
        z2 = np.dot(x, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))

    def loss_val(self):
        delta = 1e-7
        z2 = np.dot(self.xdata, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))
       
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
if __name__ == '__main__':
    (training_data, test_data) = DataGenerator("Diabetes", "./data/diabetes.csv", 0.6, True).generate()

    training_xdata = training_data[:, 0:-1]
    training_tdata = training_data[:, -1]
    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]
    
    i_node = training_xdata.shape[1]
    h1_node = 5
    o_node = 1
    lr = 1e-2
    itr_count = 5000

    # Mini-batch size == training_data size 
    # => Batch gradient descent method!
    test = Diabetes("Diabetes", training_xdata, training_tdata, i_node, h1_node, o_node, lr, itr_count)
    test.train(True, 500)

    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))


    