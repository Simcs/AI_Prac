from lib.numerical_derivative import numerical_derivative
import numpy as np

def feed_forward(xdata, tdata):
    delta = 1e-7
    z2 = np.dot(xdata, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    y = sigmoid(z3)
    return (-1) * np.sum(tdata * np.log(y + delta) + (1 - tdata) * np.log((1 - y) + delta))

def loss_val(xdata, tdata):
    delta = 1e-7
    z2 = np.dot(xdata, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    y = sigmoid(z3)
    return (-1) * np.sum(tdata * np.log(y + delta) + (1 - tdata) * np.log((1 - y) + delta))

def predict(test):
    z2 = np.dot(test, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    y = sigmoid(z3)
    if y > 0.5:
        result = 1
    else:
        result = 0
    return y, result

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':
    xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_tdata = np.array([0, 0, 0, 1]).reshape(4, 1)
    or_tdata = np.array([0, 1, 1, 1]).reshape(4, 1)
    nand_tdata = np.array([1, 1, 1, 0]).reshape(4, 1)
    xor_tdata = np.array([0, 1, 1, 0]).reshape(4, 1)

    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    input_node_num = 2
    hidden_node_num = 2
    output_node_num = 1

    W2 = np.random.rand(input_node_num, hidden_node_num)
    b2 = np.random.rand(hidden_node_num)
    W3 = np.random.rand(hidden_node_num, output_node_num)
    b3 = np.random.rand(output_node_num)

    learning_rate = 1e-2
    f = lambda x : feed_forward(xdata, xor_tdata)
    for step in range(30000):
        W2 -= learning_rate * numerical_derivative(f, W2)
        b2 -= learning_rate * numerical_derivative(f, b2)
        W3 -= learning_rate * numerical_derivative(f, W3)
        b3 -= learning_rate * numerical_derivative(f, b3)
        if step % 1000 == 0:
            print("step :", step, "loss_value :" , loss_val(xdata, xor_tdata))

    for data in test_data:
        print('predict', data, ':', predict(data))



    