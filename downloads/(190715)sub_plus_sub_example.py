#!/usr/bin/env python
# coding: utf-8

# ### 4개의 입력데이터 연산 (A1-A2+A3-A4) 예측하는 Linear Regression Batch 예제

# In[1]:


import numpy as np
from datetime import datetime

loaded_data = np.loadtxt('./regression_testdata_03.csv', delimiter=',', dtype=np.float32)

x_data = loaded_data[ :, 0:-1]
t_data = loaded_data[ :, [-1]]

# 데이터 차원 및 shape 확인
print("loaded_data.ndim = ", loaded_data.ndim, ", loaded_data.shape = ", loaded_data.shape)
print("x_data.ndim = ", x_data.ndim, ", x_data.shape = ", x_data.shape)
print("t_data.ndim = ", t_data.ndim, ", t_data.shape = ", t_data.shape) 


# In[2]:


W = np.random.rand(4,1)  # 4X1 행렬
b = np.random.rand(1)  
print("W = ", W, ", W.shape = ", W.shape, ", b = ", b, ", b.shape = ", b.shape)


# In[3]:


def loss_func(x, t):
    y = np.dot(x,W) + b
    
    return ( np.sum( (t - y)**2 ) ) / ( len(x) )


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


# 손실함수 값 계산 함수
# 입력변수 x, t : numpy type
def error_val(x, t):
    y = np.dot(x,W) + b
    
    return ( np.sum( (t - y)**2 ) ) / ( len(x) )

# 학습을 마친 후, 임의의 데이터에 대해 미래 값 예측 함수
# 입력변수 x : numpy type
def predict(x):
    y = np.dot(x,W) + b
    
    return y


# In[6]:


learning_rate = 1e-5  # 1e-2, 1e-3 은 손실함수 값 발산

f = lambda x : loss_func(x_data,t_data)

print("Initial error value = ", error_val(x_data, t_data), "Initial W = ", W, "\n", ", b = ", b )

start_time = datetime.now()

for step in  range(500001):    # 50만번 반복수행
    
    W -= learning_rate * numerical_derivative(f, W)
    
    b -= learning_rate * numerical_derivative(f, b)
    
    if (step % 5000 == 0):
        print("step = ", step, "error value = ", error_val(x_data, t_data) )
        
end_time = datetime.now()
        
print("")
print("Elapsed Time => ", end_time - start_time)


# In[7]:


ex_data_01 = np.array([4, 4, 4, 4])    #  4 - 4 + 4 - 4 = 0

print("predicted value = ", predict(ex_data_01) ) 


# In[8]:


ex_data_02 = np.array([-3, 0, 9, -1])    #  -3 -0 +9 -(-1) = 7

print("predicted value = ", predict(ex_data_02) ) 


# In[9]:


ex_data_03 = np.array([-7, -9, -2, 8])   # -7 -(-9) + (-2) -8 = -8

print("predicted value = ", predict(ex_data_03) ) 


# In[10]:


ex_data_04 = np.array([1, -2, 3, -2])   # 1 -(-2) + 3 -(-2) = 8

print("predicted value = ", predict(ex_data_04) ) 


# In[11]:


ex_data_05 = np.array([19, -12, 0, -76])   # 19 -(-12) + 0 -(-76) = 107

print("predicted value = ", predict(ex_data_05) ) 


# In[12]:


ex_data_06 = np.array([2001, -1, 109, 31])   # 2001 -(-1) + 109 -(31) = 2080

print("predicted value = ", predict(ex_data_06) ) 


# In[ ]:




