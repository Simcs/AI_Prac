import numpy as np
from datetime import datetime

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = tmp_val + delta_x
        fx1 = f(x)
        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad

class LinearRegressionTest:
    def __init__(self, xdata, tdata, learning_rate, iteration_count):
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

            # if step % 1000 == 0:
            #     print("step :", step, "error value :" , self.error_val(), "W =", self.W, "b =", self.b)
        
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

if __name__ == "__main__":
    loaded_data = np.loadtxt('data-01.csv', delimiter=',', dtype=np.float32)
    x_data = loaded_data[:, 0:-1]
    t_data = loaded_data[:, [-1]]

    test_data = np.array([100, 98, 91])

    test1 = LinearRegressionTest(x_data, t_data, 1e-5, 10000)
    test1.train()
    print("Predict", test_data, ":", test1.predict(test_data))

    test2 = LinearRegressionTest(x_data, t_data, 1e-6, 100000)
    test2.train()
    print("Predict", test_data, ":", test2.predict(test_data))
