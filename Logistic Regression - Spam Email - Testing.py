#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

df = pd.read_excel('Testing.xlsx')

# Initialize the weights and bias
w1 = 2.779
w2 = -0.205
w3 = 3.145
b = -0.084



z = (df['x1'] * w1) + (df['x2'] * w2) + (df['x3'] * w3) + b



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

predictions = sigmoid(z)

# Now we will go through each number in the list
for i in range(len(predictions)):  # "range(len(numbers))" is a way to count how many items there are in the list
    if predictions[i] > 0.5:  # If the number is bigger than 0.5
        predictions[i] = 1  # We replace it with 1
    else:  # If it's not bigger than 0.5
        predictions[i] = 0  # We replace it with 0




print(predictions)

print(df['y'])


print(4/6 * 100)

