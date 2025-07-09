'''
This has been tested to be correct!
'''

import math
import numpy as np

def dot(a, b):  # Only works on vectors!
    if (len(a) != len(b)):
        raise Exception("Vectors not same size")
    
    res = 0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res


def sigmoid(z):  # Sigma
    return 1 / (1 + (math.e)**(-z))


def loss(y_exp, y_hat):  # L
    '''y_exp = expected Y, y_pre = predicted Y'''
    return (-1)*(y_exp)*np.log(y_hat) + (1-y_exp)*np.log(1-y_hat)


def gradient_descent_step(weights, bias, m, input_data, output_data):
    J = 0
    db = 0
    dw = [0] * len(weights)

    for i in range(m):
        x = input_data[i]
        y = output_data[i]

        # Calculate loss
        z = dot(x, weights) + bias
        a = sigmoid(z)
        J += loss(y, a)
        dz = a - y
        
        # Calculate weight & bias diffs
        for j in range(len(weights)):
            dw[j] += x[j] * dz
        db += dz
    
    J /= m
    db /= m
    for j in range(len(weights)):
        dw[j] /= m

    return (J.item(), dw, db)

weights = [
    0.1, 0.2, 0.3
]
bias = 0.1
input_data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
output_data = [
    0.5,
    0.6,
    0.7
]
m = len(input_data)

l = gradient_descent_step(weights, bias, m, input_data, output_data)
print(l)