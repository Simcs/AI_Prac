import numpy as np
from lib.linear_regression import LinearRegression

if __name__ == "__main__":
    loaded_data = np.loadtxt('./data/data-01.csv', delimiter=',', dtype=np.float32)
    x_data = loaded_data[:, 0:-1]
    t_data = loaded_data[:, [-1]]

    test_data = np.array([100, 98, 81])

    test1 = LinearRegression(x_data, t_data, 1e-5, 20000)
    test1.train(True)
    print("Predict", test_data, ":", test1.predict(test_data))

    test2 = LinearRegression(x_data, t_data, 1e-6, 100000)
    test2.train(True)
    print("Predict", test_data, ":", test2.predict(test_data))
