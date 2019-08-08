import numpy as np
import pandas as pd

if __name__ == '__main__':
    train_data = pd.read_csv('./data/kaggle/titanic/train.csv').values
    print(train_data)