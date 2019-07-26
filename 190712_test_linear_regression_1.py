import numpy as np

data = np.loadtxt("data-01.csv", delimiter=",", dtype=np.float32)
x_data = data[:, 0:-1]
t_data = data[:, [-1]]

W = np.random.rand(3, 1)
b = np.random.rand(1)

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
    learning_rate = 1e-5
    f = lambda x : loss_func(x_data, t_data)
    print("initial error value :", error_val(x_data, t_data), " initial W =", W, "\nb =", b)
    for step in range(20001):
        W -= learning_rate * numerical_derivative(f, W)
        b -= learning_rate * numerical_derivative(f, b)

        if step % 1000 == 0:
            print("step :", step, "error value :", error_val(x_data, t_data), " W =", W, "\nb =", b)

    test_data = np.array([100, 98, 81])
    print('predict', test_data, ':', predict(test_data))
    