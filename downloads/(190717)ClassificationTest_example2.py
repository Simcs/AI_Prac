#!/usr/bin/env python
# coding: utf-8

# In[1]:


# x_data = (예습시간, 복습시간)
# t_data = 1 (Pass), 0 (Fail)

import numpy as np
from datetime import datetime

x_data = [ [2, 4], [4, 11], [6, 6], [8, 5], [10, 7], [12, 16], [14, 8], [16, 3], [18, 7] ]
t_data = [0, 0, 0, 0, 1, 1, 1, 1, 1]


# In[2]:


W = np.random.rand(2, 1)  # 2X1 행렬
b = np.random.rand(1)  
print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)


# In[3]:


# classification 이므로 출력함수로 sigmoid 정의

def sigmoid(x):
    return 1 / (1+np.exp(-x))


# In[4]:


# 최종출력은 y = sigmoid(Wx+b) 이며, 손실함수는 cross-entropy 로 나타냄

def loss_func(x, t):
    
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    # cross-entropy 
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) )  


# In[5]:


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


# In[6]:


def error_val(x, t):
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    # cross-entropy 
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) )  

def predict(x):
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    if y > 0.5:
        result = 1  # True
    else:
        result = 0  # False
    
    return y, result


# In[7]:


learning_rate = 1e-2  # 1e-2, 1e-3 은 손실함수 값 발산

# x_data, t_data 는 list 이므로 numpy로 바꾸어주어야 함

input_xdata = np.array(x_data)
input_tdata = np.array(t_data).reshape(len(t_data), 1)

f = lambda x : loss_func(input_xdata, input_tdata)

print("Initial error value = ", error_val(input_xdata, input_tdata), "Initial W = ", W, "\n", ", b = ", b )

start_time = datetime.now()

for step in  range(100001):  
    
    W -= learning_rate * numerical_derivative(f, W)
    
    b -= learning_rate * numerical_derivative(f, b)
    
    if (step % 1000 == 0):
        print("step = ", step, "error value = ", error_val(input_xdata, input_tdata), "W = ", W, ", b = ",b )
        
        
end_time = datetime.now()

print("")
print("Elapsed Time => ", end_time - start_time) 


# In[8]:


test_data = np.array([3, 17]) # (예습, 복습) = (3, 17) => Fail (0)
predict(test_data) 


# In[9]:


test_data = np.array([5, 8]) # (예습, 복습) = (5, 8) => Fail (0)

predict(test_data) 


# In[10]:


test_data = np.array([7, 21]) # (예습, 복습) = (7, 21) => Pass (1)

predict(test_data) 


# In[11]:


test_data = np.array([12, 0])  # (예습, 복습) = (12, 0) => Pass (1)

predict(test_data) 


# In[ ]:




