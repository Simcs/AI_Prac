import numpy as np
from lib.classification import SimpleLogisticRegression

if __name__ == "__main__":
    data = np.loadtxt("data-02.csv", delimiter=",", dtype=np.float32)
    x_data = data[:, 0:-1]
    t_data = data[:, [-1]]

    test = SimpleLogisticRegression(x_data, t_data, 1e-3, 100000, 10000)
    test.train()

    test_data1 = np.array([3, 17])
    print('predict', test_data1, ':', test.predict(test_data1))
    test_data2 = np.array([5, 8])
    print('predict', test_data2, ':', test.predict(test_data2))
    test_data3 = np.array([7, 21])
    print('predict', test_data3, ':', test.predict(test_data3))
    test_data4 = np.array([12, 0])
    print('predict', test_data4, ':', test.predict(test_data4))