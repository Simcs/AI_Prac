import numpy as np

W2 = np.array([[1, 1], [2, 1]]).astype(np.float32)
b2 = np.array([1, 1], ndmin=2).astype(np.float32)
W3 = np.array([[2, 1], [1, 1]]).astype(np.float32)
b3 = np.array([0, 2], ndmin=2).astype(np.float32)

learning_rate = 0.1
epochs = 1

def train(xdata, tdata):
    global W2, b2, W3, b3, learning_rate, epochs
    for step in range(epochs):
        for i in range(len(xdata)):
            input_data = np.array(xdata[i], ndmin=2)
            target_data = np.array(tdata[i], ndmin=2)

            # print("input", self.input_data.shape)
            z2 = np.dot(input_data, W2) + b2
            # print("z2", z2.shape)
            a2 = sigmoid(z2)
            # print("a2", a2.shape)
            z3 = np.dot(a2, W3) + b3
            # print("z3", z3.shape)
            a3 = sigmoid(z3)
            # print("a3", a3.shape)

            loss_3 = (a3 - target_data) * a3 * (1 - a3)
            W3 -= learning_rate * np.dot(a2.T, loss_3)
            b3 -= learning_rate * loss_3
            print("loss_3", loss_3)

            loss_2 = np.dot(loss_3, W3.T) * a2 * (1 - a2)
            W2 -= learning_rate * np.dot(input_data.T, loss_2)
            b2 -= learning_rate * loss_2
            print("loss_2", loss_2)

            print("W2", W2)
            print("b2", b2)
            print("W3", W3)
            print("b3", b3)
 
def sigmoid(z):
    return z - 3.

if __name__ == '__main__':
    xdata = np.array([1, 2], ndmin=2)
    tdata = np.array([2, 1], ndmin=2)
    train(xdata, tdata)

