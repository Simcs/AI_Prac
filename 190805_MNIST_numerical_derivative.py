import numpy as np
from lib.numerical_derivative import numerical_derivative

class MNIST_NumericalDerivative:
    def __init__(self, name, xdata, tdata, i_node, h1_node, o_node, lr, epochs):
        self.name = name
        if xdata.ndim == 1:     # vector
            self.xdata = xdata.reshape(len(xdata), 1)
            self.tdata = tdata.reshape(len(tdata), 1)
        else:
            self.xdata = xdata
            if tdata.ndim == 1: # vector
                self.tdata = tdata.reshape(len(tdata), 1) 
            else:
                self.tdata = tdata

        self.input_node = i_node
        self.h1_node = h1_node
        self.output_node = o_node

        # Xavier/He initialization
        self.W2 = np.random.randn(i_node, h1_node) / np.sqrt(i_node / 2)
        self.b2 = np.random.rand(1, h1_node)
        self.W3 = np.random.randn(h1_node, o_node) / np.sqrt(h1_node / 2)
        self.b3 = np.random.rand(1, o_node)

        self.learning_rate = lr
        self.epochs = epochs

    def train(self, debug=False, interval=5):
        for step in range(self.epochs):
            for i in range(len(self.xdata)):
                self.input_data = self.xdata[i]
                self.target_data = self.tdata[i]

                f = lambda x : self.feed_forward()
                self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
                self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
                self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
                self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)
                if i % 1000 == 0:
                    print("i:", i, "loss_val:", self.loss_val())
            if debug and step % interval == 0:
                print("epoch:", step, "loss_val:", self.loss_val())
    
    def predict(self, test_data):
        z2 = np.dot(test_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        result = np.argmax(y)
        return y, result
    
    def accuracy(self, xdata, tdata):
        matched_list = []
        not_matched_list = []
        index_label_prediction_list = []
        
        result = [self.predict(data)[1] for data in xdata]
        for i in range(len(result)):
            if result[i] == tdata[i]:
                matched_list.append(i)
            else:
                not_matched_list.append(i)
            index_label_prediction_list.append([i, tdata[i], result[i]])

        return matched_list, not_matched_list, index_label_prediction_list

    def feed_forward(self):
        delta = 1e-7
        z2 = np.dot(self.input_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))

    def loss_val(self):
        delta = 1e-7
        z2 = np.dot(self.input_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.target_data * np.log(y + delta) + (1 - self.target_data) * np.log((1 - y) + delta))
       
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

print('test')
training_data = np.loadtxt('./mnist_train.csv', delimiter=",", dtype=np.float32)
test_data = np.loadtxt('./mnist_test.csv', delimiter=",", dtype=np.float32)

i_node = training_data.shape[1] - 1
h1_node = 1
o_node = 10
lr = 1e-2
epochs = 1

training_xdata = (training_data[:, 1:] / 255.0) * 0.99 + 0.01
print("training_xdata:", training_xdata.shape)
training_tdata = np.zeros((training_data.shape[0], 10)) + 0.01
print("training_tdata:", training_tdata.shape)

test_xdata = (test_data[:, 1:] / 255.0) * 0.99 + 0.01
test_tdata = test_data[:, [0]]
for i in range(len(training_data)):
#     print(int(training_data[i, 0]))
    training_tdata[i, int(training_data[i, 0])] = 0.99

print(training_tdata)
test = DeepLearning("MNIST", training_xdata, training_tdata, i_node, h1_node, o_node, lr, epochs)
test.train(debug=True, interval=1)