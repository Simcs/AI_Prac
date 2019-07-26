import numpy as np
from lib.classification import SimpleLogisticRegression
from datetime import datetime

if __name__ == "__main__":
    xdata = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    tdata = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    test_data = np.array([12])

    test1 = SimpleLogisticRegression(xdata, tdata, 1e-3, 100000, 5000)
    test1.train()
    print("Predict", test_data, ":", test1.predict(test_data))