#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np

df = pd.read_excel('Training.xlsx')

# Initialize the weights and bias
w1 = 0.5
w2 = 0.01
w3 = 3
b = 0

# Define learning rate
learning_rate = 0.01

# Define the number of epochs
epochs = 1000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(epochs):
    z = (df['x1'] * w1) + (df['x2'] * w2) + (df['x3'] * w3) + b

    y_hat = sigmoid(z)

    epsilon = 1e-7  # small constant
    loss_function = - (df['y'] * np.log(y_hat + epsilon) + (1 - df['y']) * np.log(1 - y_hat + epsilon))

    cost_function = np.mean(loss_function)

    # Backpropagation
    dz = y_hat - df['y']

    dw1 = np.mean(dz * df['x1'])
    dw2 = np.mean(dz * df['x2'])
    dw3 = np.mean(dz * df['x3'])
    db = np.mean(dz)

    # Update weights and bias
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    w3 = w3 - learning_rate * dw3
    b = b - learning_rate * db

    # Print the epoch number, cost function and updated weights and bias
    print(f"Epoch: {i+1}, Cost_function: {cost_function}, w1: {w1}, w2: {w2}, w3: {w3}, b: {b}")


# In[ ]:




