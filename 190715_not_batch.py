import numpy as np
import time

x_data = np.array([1, 2, 3, 4, 5])
t_data = np.array([3, 4, 5, 6, 7])

W = np.random.rand(1, 1)
b = np.random.rand(1)
print('W = ', W , "W shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)

import numpy as np

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
    learning_rate = 1e-2
    start_time = time.time()
    
    for step in range(8001):
        for idx in range(len(x_data)):
            input_x = x_data[idx]
            input_t = t_data[idx]
            f = lambda x : loss_func(np.array([input_x]), np.array([input_t]))
            W -= learning_rate * numerical_derivative(f, W)
            b -= learning_rate * numerical_derivative(f, b)

        if step % 400 == 0:
            print("step :", step, "error value :" , error_val(x_data.reshape(5, 1), t_data.reshape(5, 1)), "W = ", W, "b = ", b)
            
    elapsed_time = time.time() - start_time
    print('elapsed time : ', elapsed_time)

    test_score = np.array([44])
    print('predict', test_score, ':', predict(test_score))
    