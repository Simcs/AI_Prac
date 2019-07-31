import numpy as np
from NumericalDerivative import numerical_derivative
from datetime import datetime

class LinearRegressionTest:
    def __init__(self, xdata, tdata, learning_rate, iteration_count):
        if xdata.ndim == 1:     # vector
            self.xdata = xdata.T
            self.tdata = tdata.T
        elif xdata.ndim == 2:   # matrix
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

    def train(self):
        f = lambda x : self.loss_func()
        print("initial error value :", self.error_val(), "initial W =", self.W, "\nb =", self.b)
        start_time = datetime.now()

        for step in range(self.iteration_count + 1):
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
            if step % 1000 == 0:
                print("step :", step, "error value :" , self.error_val(), "W =", self.W, "b =", self.b)
        
        print("final error value :", self.error_val(), "final W =", self.W, "\nb =", self.b)
        print("Elapsed time :", datetime.now() - start_time)

    def predict(self, testdata):
        y = np.dot(testdata, self.W) + self.b
        return y

    def loss_func(self):
        y = np.dot(self.xdata, self.W) + self.b
        return np.sum((self.tdata - y) ** 2) / len(self.xdata)

    def error_val(self):
        y = np.dot(self.xdata, self.W) + self.b
        return np.sum((self.tdata - y) ** 2) / len(self.xdata)