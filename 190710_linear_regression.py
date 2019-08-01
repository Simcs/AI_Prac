import numpy as np
from lib.numerical_derivative import numerical_derivative

x_data = np.array([1, 2, 3, 4, 5, 20, 400]).reshape(7, 1)
t_data = np.array([1, 4, 9, 16, 25, 400, 160000]).reshape(7, 1)

W = np.random.rand(1, 1)
b = np.random.rand(1)
print('W = ', W , "W shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

# function which calculates y and E(W, b)
def loss_func(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y) ** 2) / len(x)

# error_val does exactly what loss_func function does.
def error_val(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y) ** 2) / len(x)

def predict(x):
    y = np.dot(x, W) + b
    return y

if __name__ == "__main__":
    learning_rate = 1e-2
    f = lambda x : loss_func(x_data, t_data)
    print("initial error value : ", error_val(x_data, t_data), " initial W  = ", W, "\nb = ", b)
    for step in range(8001):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate * numerical_derivative(f, b)

        if step % 400 == 0:
            print("step : ", step, "error value : " , error_val(x_data, t_data), " W = ", W, " b = ", b)

    test_score = np.array([44])
    print('predict', test_score, ':', predict(test_score))
    