#!/usr/bin/env -S uv run

import numpy as np

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
    
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    
    # unvectorized implementation (vectorization is covered later).
    m = x.shape[0]

    cost_acc = 0
    for i in range(m):
        err = y[i] - (w * x[i] + b)
        cost_acc = cost_acc + err ** 2

    return cost_acc / (2 * m)

#x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
#y_train = np.array([300.0, 500.0])       #(price in 1000s of dollars)

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

w = 209
b = 2.4

print(f"{compute_cost(x_train, y_train, w, b)}")

