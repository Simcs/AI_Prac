from lib.data_generator import DataGenerator
import numpy as np

class BackPropagation:
    def __init__(self, name, xdata, tdata, i_node, h1_node, h2_node, o_node, lr, epochs):
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
        
        self.W2 = np.random.randn(i_node, h1_node) / np.sqrt(i_node / 2)
        self.b2 = np.random.rand(1, h1_node)
        self.W3 = np.random.randn(h1_node, h2_node) / np.sqrt(h1_node / 2)
        self.b3 = np.random.rand(1, h2_node)
        self.W4 = np.random.rand(h2_node, o_node) / np.sqrt(h2_node / 2)
        self.b4 = np.random.rand(1, o_node)

        self.learning_rate = lr
        self.epochs = epochs
    
    def train(self, debug=False, interval=100):
        for step in range(self.epochs):
            for i in range(len(self.xdata)):
                self.input_data = np.array(self.xdata[i], ndmin=2)
                self.target_data = np.array(self.tdata[i], ndmin=2)
                self.feed_forward()

                loss_4 = (self.A4 - self.target_data) * self.A4 * (1 - self.A4)
                self.W4 -= self.learning_rate * np.dot(self.A3.T, loss_4)
                self.b4 -= self.learning_rate * loss_4

                loss_3 = np.dot(loss_4, self.W4.T) * self.A3 * (1 - self.A3)
                self.W3 -= self.learning_rate * np.dot(self.A2.T, loss_3)
                self.b3 -= self.learning_rate * loss_3

                loss_2 = np.dot(loss_3, self.W3.T) * self.A2 * (1 - self.A2)
                self.W2 -= self.learning_rate * np.dot(self.A1.T, loss_2)
                self.b2 -= self.learning_rate * loss_2
            if debug and step % interval == 0:
                print("epoch:", step, "loss_val:", self.loss_val())
                
    def feed_forward(self):
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.sigmoid(self.Z4)

    def loss_val(self):
        self.Z1 = self.input_data
        self.A1 = self.input_data

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.sigmoid(self.Z3)

        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.A4 = self.sigmoid(self.Z4)

        delta = 1e-7
        return (-1) * np.sum(self.target_data * np.log(self.A4 + delta) + (1 - self.target_data) * np.log((1 - self.A4) + delta))

    def predict(self, input_data):
        
        Z2 = np.dot(input_data, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        Z3 = np.dot(A2, self.W3) + self.b3
        A3 = self.sigmoid(Z3)
        Z4 = np.dot(A3, self.W4) + self.b4
        y = self.sigmoid(Z4)
        
        if y >= 0.5:
            predicted_num = 1
        else:
            predicted_num = 0
        return y, predicted_num

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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

if __name__ == "__main__":
    (training_data, test_data) = DataGenerator("ThoracicSurgery", "./data/ThoracicSurgery.csv", 0.4, True).generate()
    training_xdata = training_data[:, 0:-1]
    training_tdata = training_data[:, [-1]]

    i_node = training_xdata.shape[1]
    h1_node = 10
    h2_node = 10
    o_node = 1
    lr = 1e-2
    epochs = 50

    test = BackPropagation("ThoracicSurgery", training_xdata, training_tdata, i_node, h1_node, h2_node, o_node, lr, epochs)
    test.train(debug=True, interval=10)

    test_xdata = test_data[:, 0:-1]
    test_tdata = test_data[:, -1]
    (matched_list, not_mathced_list, prediction_list) = test.accuracy(test_xdata, test_tdata)
    print(prediction_list)
    print("accuracy:", len(matched_list) / len(test_xdata))

