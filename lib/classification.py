import numpy as np
from numerical_derivative import *
from datetime import datetime

class SimpleLogisticRegression:
    def __init__(self, xdata, tdata, learning_rate, iteration_count, debug=False, interval=None):
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
        self.debug = debug
        if interval is not None:
            self.interval = interval
        else:
            self.interval = 1000

        self.W = np.random.rand(self.xdata.shape[1], 1)
        self.b = np.random.rand(1)

    def getW(self):
        return self.W
    
    def getB(self):
        return self.b

    def setInterval(self, interval):
        self.interval = interval

    def train(self):
        f = lambda x : self.loss_func()
        start_time = datetime.now()
        
        if self.debug:
            print("initial error value :", self.error_val(), "initial W =", self.W, "\nb =", self.b)

        for step in range(self.iteration_count + 1):
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
            if self.debug and step % self.interval == 0:
                print("step :", step, "error value :" , self.error_val(), "W =", self.W, "b =", self.b)
        
        if self.debug:
            print("final error value :", self.error_val(), "final W =", self.W, "\nb =", self.b)
            print("Elapsed time :", datetime.now() - start_time)

    def accuracy(self, xdata, tdata=None):
        if tdata is None:
            xdata = xdata[:, 0:-1]
            tdata = xdata[:, [-1]]
        else:
            if tdata.ndim == 1:
                tdata = tdata.reshape(len(tdata), 1)

        accurate_num = 0
        result = [self.predict(data)[1] for data in xdata]
        for i in range(len(result)):
            if result[i] == tdata[i]:
                accurate_num += 1
        return accurate_num / len(xdata)

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
        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def error_val(self):
        delta = 1e-7
        z = np.dot(self.xdata, self.W) + self.b
        y = self.sigmoid(z)
        return -np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))