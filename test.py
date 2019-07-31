import numpy as np
import random

arr1 = [[2, 3, 4, 5, 6],
        [4, 6, 8, 10, 12],
        [6, 9, 12, 15, 18],
        [8, 12, 16, 20, 24]]
arr2 = [2, 3, 4, 5, 6]

numpy_arr = np.array(arr1)
np.random.shuffle(arr1) 
np.random.shuffle(numpy_arr)

print( arr1 )
print( np.divide(arr1 , arr2) )
print(numpy_arr)