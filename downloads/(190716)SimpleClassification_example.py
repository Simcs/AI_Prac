#!/usr/bin/env python
# coding: utf-8

# ## Simple Classification 구현
# ### 주의해서 볼 함수는 sigmoid,  loss_func,  predict 함수

# In[1]:


import numpy as np
from datetime import datetime

x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10,1)   
t_data = np.array([0, 0, 0, 0,  0,  0,  1,  1,  1,  1]).reshape(10,1)

print("x_data.shape = ", x_data.shape, ", t_data.shape = ", t_data.shape)


# In[2]:


W = np.random.rand(1,1)  
b = np.random.rand(1)  
print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)


# In[3]:


# 최종출력은 y = sigmoid(Wx+b) 이며, 손실함수는 cross-entropy 로 나타냄

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def loss_func(x, t):
    
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    # cross-entropy 
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) ) 

# 손실함수 값 계산 함수
# 입력변수 x, t : numpy type
def error_val(x, t):
    delta = 1e-7    # log 무한대 발산 방지
    
    z = np.dot(x, W) + b
    y = sigmoid(z)
    
    # cross-entropy 
    return  -np.sum( t*np.log(y + delta) + (1-t)*np.log((1 - y)+delta ) ) 

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 test_data : numpy type
def predict(test_data):
    
    z = np.dot(test_data, W) + b
    y = sigmoid(z)
    
    if y >= 0.5:
        result = 1  # True
    else:
        result = 0  # False
    
    return y, result


# In[4]:


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


# In[5]:


learning_rate = 1e-2  # 발산하는 경우, 1e-3 ~ 1e-6 등으로 바꾸어서 실행

f = lambda x : loss_func(x_data,t_data)  # f(x) = loss_func(x_data, t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", W, "\n", ", b = ", b )

start_time = datetime.now()

for step in  range(400001):  
    
    W -= learning_rate * numerical_derivative(f, W)
    
    b -= learning_rate * numerical_derivative(f, b)
    
    if (step % 5000 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data), "W = ", W, ", b = ",b )
        
end_time = datetime.now()
        
print("")
print("Elapsed Time => ", end_time - start_time)


# In[6]:


test_data = np.array([3.7])

(real_val, logical_val) = predict(test_data)

print(real_val, logical_val)


# In[7]:


test_data = np.array([31.09])

(real_val, logical_val) = predict(test_data)

print(real_val, logical_val)


# In[ ]:




