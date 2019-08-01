import numpy as np
from lib.numerical_derivative import *
from datetime import datetime

class SimpleLogisticRegression:
    def __init__(self, xdata, tdata, learning_rate, iteration_count):
        if xdata.ndim == 1:     # vector
            self.xdata = xdata.reshape(len(xdata), 1)
            self.tdata = tdata.reshape(len(tdata), 1)
        elif xdata.ndim == 2 and tdata.ndim == 1:   # matrix / vector
            self.xdata = xdata
            self.tdata = tdata.reshape(len(tdata), 1) # vector
        elif xdata.ndim == 2 and tdata.ndim == 2:
            self.xdata = xdata
            self.tdata = tdata

        self.learning_rate = learning_rate
        self.iteration_count = iteration_count

        self.W = np.random.rand(self.xdata.shape[1], 1)
        self.b = np.random.rand(1)

    def getW(self):
        return self.W
    
    def getB(self):
        return self.b

    def train(self, debug=True, interval=1000):
        f = lambda x : self.loss_func()
        start_time = datetime.now()
        
        if debug:
            print("initial error value :", self.error_val(), "initial W =", self.W, "\nb =", self.b)

        for step in range(self.iteration_count + 1):
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
            if debug and step % interval == 0:
                print("step :", step, "error value :" , self.error_val(), "W =", self.W, "b =", self.b)
        
        if debug:
            print("final error value :", self.error_val(), "final W =", self.W, "\nb =", self.b)
            print("Elapsed time :", datetime.now() - start_time)

    def accuracy1(self, xdata, tdata):
        if tdata.ndim == 1:
            tdata = tdata.reshape(len(tdata), 1)

        accurate_num = 0
        result = [self.predict(data)[1] for data in xdata]
        for i in range(len(result)):
            if result[i] == tdata[i]:
                accurate_num += 1
        return accurate_num / len(xdata)

    def accuracy2(self, test_data):
        xdata = test_data[:, 0:-1]
        tdata = test_data[:, [-1]]
        return self.accuracy1(xdata, tdata)

    def predict(self, testdata):
        z = np.dot(testdata, self.W) + self.b
        y = self.sigmoid(z)
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

    def loss_func(self):
        delta = 1e-7
        z = np.dot(self.xdata, self.W) + self.b
        y = self.sigmoid(z)
        return (-1) * np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def error_val(self):
        delta = 1e-7
        z = np.dot(self.xdata, self.W) + self.b
        y = self.sigmoid(z)
        return (-1) * np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))