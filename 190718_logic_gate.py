import numpy as np
from lib.classification import SimpleLogisticRegression
from lib.numerical_derivative import *
from datetime import datetime

class LogicGate:
    def __init__(self, xdata, tdata, learning_rate, iteration_count, interval=None):
        if xdata.ndim == 1:
            self.xdata = xdata.reshape(len(xdata), 1)
            self.tdata = tdata.reshape(len(tdata), 1)
        elif xdata.ndim == 2:
            self.xdata = xdata
            if tdata.ndim == 1:
                self.tdata = tdata.reshape(len(tdata), 1)
            else:
                self.tdata = tdata

        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
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
        for step in range(self.iteration_count + 1):
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
            if step % self.interval == 0:
                print('step:', step, 'error_val:', self.error_val(), 'W:', self.W, '\nb:', self.b)
        elapsed_time = datetime.now() - start_time
        print('final error value', self.error_val(), 'W:', self.W, '\nb:', self.b)
        print('elapsed time:', elapsed_time)

    def predict(self, test_data):
        z = np.dot(test_data, self.W) + self.b
        y = self.sigmoid(z)
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

    def accuracy(self, data1, data2=None):
        if data2 is None:
            xdata = data1[:, 0:-1]
            tdata = data1[:, [-1]]
        else:
            xdata = data1
            if data2.ndim == 1:
                tdata = data2.reshape(len(data2), 1)
            else:
                tdata = data2

        accurate_num = 0
        result = [self.predict(data)[1] for data in xdata]
        for i in range(len(result)):
            if result[i] == tdata[i]:
                accurate_num += 1
        return accurate_num / len(xdata)

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

if __name__ == "__main__":
    test_xdata = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
    test_tdata = np.array([1, 0, 0, 1])
    test = SimpleLogisticRegression(test_xdata, test_tdata, 1e-3, 10000, 5000)
    test.train()
    # print('accuracy:', test.accuracy(test_xdata, test_tdata))
    test_data = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    print('accuracy:', test.accuracy(test_data))