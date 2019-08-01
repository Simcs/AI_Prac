import numpy as np
from lib.classification import SimpleLogisticRegression
from datetime import datetime

if __name__ == "__main__":
    test_xdata = np.array([[0, 0], [0, 1], [1, 0],[1, 1]])
    test_tdata = np.array([1, 0, 0, 1])

    test = SimpleLogisticRegression(test_xdata, test_tdata, 1e-3, 10000)
    test.train(debug=True, interval=5000)

    # accuracy version 1
    print('accuracy:', test.accuracy1(test_xdata, test_tdata))

    # accuracy version 2
    test_data = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    print('accuracy:', test.accuracy2(test_data))