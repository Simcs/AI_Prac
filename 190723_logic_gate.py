import numpy as np
from lib.numerical_derivative import *
from datetime import datetime

class LogicGate:
    def __init__(self, name, xdata, tdata, i_node, h1_node, o_node, learning_rate, iteration_count):
        self.name = name
        self.xdata = xdata
        self.tdata = tdata.reshape(len(tdata), 1)
        
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count

        self.W2 = np.random.rand(i_node, h1_node)
        self.b2 = np.random.rand(h1_node)
        self.W3 = np.random.rand(h1_node, o_node)
        self.b3 = np.random.rand(o_node)
    
    def train(self, debug=False, interval=1000):
        f = lambda x : self.feed_forward()
        start_time = datetime.now()
        
        if debug:
            print(self.name, "gate initial loss value :", self.loss_val())

        for step in range(self.iteration_count + 1):
            self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
            self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)
            self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
            self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)
            if debug and step % interval == 0:
                print(self.name, "gate step :", step, "loss_value :" , self.loss_val())
        
        if debug:
            print(self.name, 'gate train result:')
            print("final loss value :", self.loss_val())
            print("final W2 =", self.W2)
            print("final b2 =", self.b2)
            print("final W3 =", self.W3)
            print("final b3 =", self.b3)
            print(self.name, "gate Elapsed time :", datetime.now() - start_time)

    def predict(self, test_data):
        z2 = np.dot(test_data, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        if y > 0.5:
            result = 1
        else:
            result = 0
        return y, result

    def accuracy1(self, xdata, tdata):
        if tdata.ndim == 1:
            tdata = tdata.reshape(len(tdata), 1)

        accurate_num = 0
        result = [self.predict(data)[1] for data in xdata]
        for i in range(len(result)):
            if result[i] == tdata[i]:
                accurate_num += 1
        return accurate_num / len(xdata)

    def accuracy2(self, test_data):
        xdata = test_data[:, 0:-1]
        tdata = test_data[:, [-1]]
        return self.accuracy1(xdata, tdata)

    def feed_forward(self):
        delta = 1e-7
        z2 = np.dot(xdata, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))
    
    def loss_val(self):
        delta = 1e-7
        z2 = np.dot(xdata, self.W2) + self.b2
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        y = self.sigmoid(z3)
        return (-1) * np.sum(self.tdata * np.log(y + delta) + (1 - self.tdata) * np.log((1 - y) + delta))
       
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    xdata = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])

    input_node = 2
    h1_node = 4
    output_node = 1

    # # TEST_AND_GATE
    # and_tdata = np.array([0, 0, 0, 1])
    # test_and_data = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    # gate = LogicGate("AND", xdata, and_tdata, input_node, h1_node, output_node, 1e-1, 10000)
    # gate.train(debug=True, interval=1000)
    # print(gate.name, 'accuracy1:', gate.accuracy1(xdata, and_tdata))
    # print(gate.name, 'accuracy2:', gate.accuracy2(test_and_data))
    
    # # TEST_OR_GATE
    # or_tdata = np.array([0, 1, 1, 1])
    # test_or_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # gate = LogicGate("OR", xdata, or_tdata, input_node, h1_node, output_node, 1e-1, 10000)
    # gate.train(debug=True, interval=1000)
    # print(gate.name, 'accuracy1:', gate.accuracy1(xdata, or_tdata))
    # print(gate.name, 'accuracy2:', gate.accuracy2(test_or_data))

    # # TEST_NAND_GATE
    # nand_tdata = np.array([1, 1, 1, 0])
    # test_nand_data = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    # gate = LogicGate("NAND", xdata, nand_tdata, input_node, h1_node, output_node, 1e-1, 10000)
    # gate.train(debug=True, interval=1000)
    # print(gate.name, 'accuracy1:', gate.accuracy1(xdata, nand_tdata))
    # print(gate.name, 'accuracy2:', gate.accuracy2(test_nand_data))

    # TEST_XOR_GATE
    xor_tdata = np.array([0, 1, 1, 0])
    test_xor_data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    gate = LogicGate("XOR", xdata, xor_tdata, input_node, h1_node, output_node, 1e-2, 30000)
    gate.train(debug=True, interval=1000)
    print(gate.name, 'accuracy1:', gate.accuracy1(xdata, xor_tdata))
    print(gate.name, 'accuracy2:', gate.accuracy2(test_xor_data))