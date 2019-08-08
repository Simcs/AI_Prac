import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# training_data = np.loadtxt('./data/mnist_train.csv', delimiter=",", dtype=np.float32)
# test_data = np.loadtxt('./data/mnist_test.csv', delimiter=",", dtype=np.float32)

# img = training_data[1901, 1:].reshape(28, 28)

# plt.imshow(img, cmap='gray')
# plt.show()

img = np.array(Image.open("./data/Sample_9.png").convert('L')).reshape(1, 784)
