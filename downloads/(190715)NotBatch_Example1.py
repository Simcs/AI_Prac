#!/usr/bin/env python
# coding: utf-8

# ### batch 사용하지 않고 입력데이터 하나씩 대입하는 예제

# In[1]:


import numpy as np

x_data = np.array([1, 2, 3, 4, 5])
t_data = np.array([2, 3, 4, 5, 6])

print("x_data.shape = ", x_data.shape, ", t_data.shape = ", t_data.shape)


# In[2]:


W = np.random.rand(1,1)  
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


learning_rate = 1e-2  

print("Initial error value = ", error_val(x_data.reshape(5,1), t_data.reshape(5,1)), "Initial W = ", W, "\n", ", b = ", b )

for step in  range(8001):  
    
    for index in range(len(x_data)):
        
        input_x_data = x_data[index]
        input_t_data = t_data[index]
        
        # np.array() 해주지 않으면 error => len() 함수를 쓰기위해서는 numpy 객체여야 하는데
        # np.array() 해주지 않으면 scala 값이기 때문에 에러 발생
        f = lambda x : loss_func(np.array([input_x_data]), np.array([input_t_data]))  
    
        W -= learning_rate * numerical_derivative(f, W)
    
        b -= learning_rate * numerical_derivative(f, b)
    
    if (step % 400 == 0):
        print("step = ", step, "error value = ", error_val(x_data.reshape(5,1), t_data.reshape(5,1)))
        
print("final W = ", W, ", and b = ", b)


# In[7]:


print("predicted value = ", predict(43)) 


# In[ ]:




