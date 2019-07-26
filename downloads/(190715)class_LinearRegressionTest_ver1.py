#!/usr/bin/env python
# coding: utf-8

# ## Class 를 이용하여 data-01.csv 선형회귀 구현

# In[1]:


import numpy as np
from datetime import datetime


class LinearRegressionTest:
    
    # constructor
    def __init__(self, xdata, tdata, learning_rate, iteration_count):
        
        self.xdata = xdata
        self.tdata = tdata
        
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        
        self.W = np.random.rand(self.xdata.shape[1], 1)   # 입력 xdata가 이미 행렬이라 가정한 구현
        self.b = np.random.rand(1)
        
        print("LinearRegressionTest Object is created")
        
    
    # obtain current W and current b
    def getW_b(self):
        
        return self.W, self.b
    
    
    # loss function
    def loss_func(self):
        
        y = np.dot(self.xdata, self.W) + self.b
    
        return ( np.sum( (self.tdata - y)**2 ) ) / ( len(self.xdata) )
        
    
    # display current error value
    def error_val(self):
        
        y = np.dot(self.xdata, self.W) + self.b
    
        return ( np.sum( (self.tdata - y)**2 ) ) / ( len(self.xdata) )
    
    
    # predict method
    def predict(self, test_data):
        
        y = np.dot(test_data, self.W) + self.b
        
        return y
    
    
    # train method
    def train(self):
    
        f = lambda x : self.loss_func()

        print("Initial error value = ", self.error_val(), "Initial W = ", self.W, "\n", ", b = ", self.b )

        start_time = datetime.now()
        
        for step in  range(self.iteration_count):  
    
            self.W -= self.learning_rate * numerical_derivative(f, self.W)
    
            self.b -= self.learning_rate * numerical_derivative(f, self.b)
    
            if (step % 400 == 0):
                print("step = ", step, "error value = ", self.error_val(), "W = ", self.W, ", b = ", self.b )
                
        end_time = datetime.now()
        
        print("")
        print("Elapsed Time => ", end_time - start_time)


# In[2]:


def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    
    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)
        
        x[idx] = tmp_val - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)
        
        x[idx] = tmp_val 
        it.iternext()   
        
    return grad


# In[3]:


loaded_data = np.loadtxt('./data-01.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[ :, 0:-1]
t_data = loaded_data[ :, [-1]]

# 데이터 차원 및 shape 확인
print("x_data.ndim = ", x_data.ndim, ", x_data.shape = ", x_data.shape)
print("t_data.ndim = ", t_data.ndim, ", t_data.shape = ", t_data.shape) 


# In[4]:


# LinearRegressionTest 객체를 만들기 위해 4개의 파라미터 필요
# 1st : 입력데이터,  2nd : 정답데이터
# 3rd : learning rate,  4th : iteration count
obj = LinearRegressionTest(x_data, t_data, 1e-5, 10001)

obj.train()


# In[5]:


test_data = np.array([100, 98, 81])

obj.predict(test_data) 


# In[6]:


(W, b) = obj.getW_b()

print("W = ", W, ", b = ", b)

