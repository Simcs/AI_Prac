from lib.classification import SimpleLogisticRegression
import numpy as np

if __name__ == "__main__":
    xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tdata = np.array([0, 0, 0, 1])
    test1 = SimpleLogisticRegression(xdata, tdata, 1e-3, 10000)
    test1.train(debug=True, interval=1000)

    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for data in test_data:
        print('predict', data, ':', test1.predict(data))

    # XOR problem!
    xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    tdata = np.array([0, 1, 1, 0])
    test2 = SimpleLogisticRegression(xdata, tdata, 1e-2, 10000)
    test2.train(debug=True, interval=1000)

    test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for data in test_data:
        print('predict', data, ':', test2.predict(data))