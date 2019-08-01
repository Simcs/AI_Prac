import numpy as np
from lib.numerical_derivative import numerical_derivative

data = np.loadtxt("./data/regression_testdata_03.csv", delimiter=",", dtype=np.float32)
x_data = data[:, 0:-1]
t_data = data[:, [-1]]

W = np.random.rand(len(x_data[0]), 1)
b = np.random.rand(1)

# function which calculates y and E(W, b)
def loss_func(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y) ** 2) / len(x)

# error_val does exactly same as what loss_func function does.
def error_val(x, t):
    y = np.dot(x, W) + b
    return np.sum((t - y) ** 2) / len(x)

def predict(x):
    y = np.dot(x, W) + b
    return y

if __name__ == "__main__":
    # 1e-2 ~ 1e-8
    learning_rate = 1e-5
    f = lambda x : loss_func(x_data, t_data)
    print("initial error value :", error_val(x_data, t_data), "W =", W, "\nb =", b)
    # 8000 ~ 500000
    for step in range(20001):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate * numerical_derivative(f, b)

        if step % 10000 == 0:
            print("step :", step, "error value :", error_val(x_data, t_data), " W =", W, "\nb =", b)

    print("final error value :", error_val(x_data, t_data), " W =", W, "\nb =", b)
    test_data = [[4, 4, 4, 4], [-3, 0, 9, -1], [7, -9, 2, -8]]
    for data in test_data:
        print('predict', data, ':', predict(np.array(data)))