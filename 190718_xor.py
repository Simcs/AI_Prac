import numpy as np
from lib.classification import SimpleLogisticRegression

if __name__ == "__main__":
    data_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    data_t_nand = np.array([1, 1, 1, 0])
    data_t_or = np.array([0, 1, 1, 1])
    data_t_and = np.array([0, 0, 0, 1])

    test_nand = SimpleLogisticRegression(data_x, data_t_nand, 1e-3, 10000)
    test_nand.train()
    test_or = SimpleLogisticRegression(data_x, data_t_or, 1e-3, 10000)
    test_or.train()
    test_and = SimpleLogisticRegression(data_x, data_t_and, 1e-3, 10000)
    test_and.train()

    print('nand accuracy:', test_nand.accuracy(data_x, data_t_nand))
    print('or accuracy:', test_or.accuracy(data_x, data_t_or))
    print('and accuracy:', test_and.accuracy(data_x, data_t_and))

    input_data = data_x
    output_data = []
    for x in data_x:
        s1 = test_nand.predict(x)[1]
        s2 = test_or.predict(x)[1]
        output_data.append(test_and.predict(np.array([s1, s2]))[1])
    
    print('xor result')
    for i in range(len(data_x)):
        print('input:', input_data[i], 'output:', output_data[i])