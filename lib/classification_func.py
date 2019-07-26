import numpy as np
from NumericalDerivative import numerical_derivative
from datetime import datetime

def train(learning_rate, iteration_count, interval):
    f = lambda x : loss_func()
    print("initial error value :", error_val(), "initial W =", W, "\nb =", b)
    start_time = datetime.now()

    for step in range(iteration_count + 1):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate * numerical_derivative(f, b)
        if step % interval == 0:
            print("step :", step, "error value :" , error_val(), "W =", W, "b =", b)
        
    print("final error value :", error_val(), "final W =", W, "\nb =", b)
    print("Elapsed time :", datetime.now() - start_time)

def predict(testdata, W, b):
    z = np.dot(testdata, W) + b
    y = sigmoid(z)
    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

def loss_func():
    delta = 1e-7
    z = np.dot(xdata, W) + b
    y = sigmoid(z)
    return -np.sum(tdata * np.log(y + delta) + (1 - tdata) * np.log((1 - y) + delta))

def error_val(xdata):
    delta = 1e-7
    z = np.dot(xdata, W) + b
    y = sigmoid(z)
    return -np.sum(tdata * np.log(y + delta) + (1 - tdata) * np.log((1 - y) + delta))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))