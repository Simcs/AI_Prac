#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from datetime import datetime

class SimpleClassificationTest:
    
    # constructor
    def __init__(self, xdata, tdata, learning_rate, iteration_count):
            
        # 가중치 W 형상을 자동으로 구하기 위해 입력데이터가 vector 인지,
        # 아니면 matrix 인지 체크 후, 
        # self.xdata 는 무조건 matrix 로 만들어 주면 코드 일관성이 있음
        
        if xdata.ndim == 1:    # vector
            self.xdata = xdata.reshape(len(xdata), 1)
            self.tdata = xdata.reshape(len(tdata), 1)
            
        elif xdata.ndim == 2:  # matrix
            self.xdata = xdata
            self.tdata = tdata
        
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count
        
        self.W = np.random.rand(self.xdata.shape[1], 1) 
        self.b = np.random.rand(1)
        
        print("SimpleClassificationTest Object is created")
        
    
    def sigmoid(self, z):
        
        return 1 / (1+np.exp(-z))
        
    # obtain current W and current b
    def getW_b(self):
        
        return self.W, self.b
    
    
    # loss function
    def loss_func(self):
        
        delta = 1e-7    # log 무한대 발산 방지
    
        z = np.dot(self.xdata, self.W) + self.b
        
        y = self.sigmoid(z)
    
        # cross-entropy 
        return  -np.sum( self.tdata*np.log(y + delta) + (1-self.tdata)*np.log((1 - y)+delta ) ) 
        
    
    # display current error value
    def error_val(self):
        
        delta = 1e-7    # log 무한대 발산 방지
    
        z = np.dot(self.xdata, self.W) + self.b
        
        y = self.sigmoid(z)
    
        # cross-entropy 
        return  -np.sum( self.tdata*np.log(y + delta) + (1-self.tdata)*np.log((1 - y)+delta ) ) 
    
    
    # predict method
    # 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
    # 입력변수 x : numpy type
    def predict(self, test_data):
    
        z = np.dot(test_data, self.W) + self.b
        y = self.sigmoid(z)
    
        if y >= 0.5:
            result = 1  # True
        else:
            result = 0  # False
    
        return y, result
    
    
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


# 입력데이터 / 정답데이터 세팅

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10,1)   
t_data = np.array([0, 0, 0, 0,  0,  0,  1,  1,  1,  1]).reshape(10,1)

print("x_data.shape = ", x_data.shape, ", t_data.shape = ", t_data.shape)


# ### learning_rate = 1e-2,  반복횟수 400,000번 수행하는 obj1

# In[4]:


obj1 = SimpleClassificationTest(x_data, t_data, 1e-2, 400001)

obj1.train()


# In[5]:


test_data = np.array([3.7])

(real_val, logical_val) = obj1.predict(test_data)

print(real_val, logical_val)


# In[6]:


test_data = np.array([31.09])

(real_val, logical_val) = obj1.predict(test_data)

print(real_val, logical_val)


# In[ ]:




