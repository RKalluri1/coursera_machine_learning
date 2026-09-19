#!/usr/bin/env -S uv run

import math, copy
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
        err = (w * x[i] + b) - y[i] 
        cost_acc = cost_acc + err ** 2

    return cost_acc / (2 * m)

def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # unvectorized implementation (vectorization is covered later).
    m = x.shape[0]

    # for linear regression, dj_dw is the sum of ((y - (w * x + b)) * x) 
    # and dj_db is the sum of (y - (w * x + b)) over all corresponding x and y.

    dj_dw = 0
    dj_db = 0
    for i in range(m):
        err = (w * x[i] + b) - y[i] 
        dj_dw = dj_dw + (err * x[i])
        dj_db = dj_db + (err)

    return dj_dw/m, dj_db/m

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
    """

    # for each iteration, get the gradient for w and b and apply
    # the gradient and learning rate to calculate the new w and b.

    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        cost = cost_function(x, y, w, b)

        if i% math.ceil(num_iters/100) == 0:    
            print(f"iteration {i}, cost={cost:0.2e}, w={w:0.3e}, b={b:0.5e}")

    return w, b, [], []

x_train = np.array([1.0, 2.0])           # features
y_train = np.array([300.0, 500.0])       # target value

#x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
#y_train = np.array([250, 300, 480,  430,   630, 730,])

w = 0
b = 0
alpha = 1e-2

w_calc, b_calc, jhist, phist = gradient_descent(x_train, y_train, w, b, alpha, 10000, compute_cost, compute_gradient)

print(f"w = {w_calc}, b = {b_calc}")

