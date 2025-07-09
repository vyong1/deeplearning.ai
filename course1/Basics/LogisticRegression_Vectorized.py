import math
import numpy as np

def sigmoid(z):  # Sigma
    return 1 / (1 + (math.e)**(-z))


def loss(y_exp, y_hat):  # L
    '''y_exp = expected Y, y_pre = predicted Y'''
    return (-1)*(y_exp)*np.log(y_hat) + (1-y_exp)*np.log(1-y_hat)


def gradient_descent_step(weights, bias, m, n, X, Y):
    '''
    Args:
        m (int): Number of data points
        n (int): Number of features
        X (arr): Inputs
        Y (arr): Outputs
    '''
    # Step data
    L = np.zeros((m, 1))
    z = np.zeros((m, 1))
    a = np.zeros((m, 1))
    dz = np.zeros((m, 1))

    # Cumulative data
    J = 0
    db = 0
    dw = np.zeros((n, 1))

    for i in range(m):
        # Calculate loss
        z[i] = np.dot(X[i], weights) + bias
        a[i] = sigmoid(z[i])
        L[i] = loss(Y[i], a[i])
        dz[i] = a[i] - Y[i]
        
        # Accumulate
        J += L[i]
        db += dz[i]
    
    # Vectorized dw calculation
    dw = np.transpose(X).dot(dz)

    # Mean
    J /= m
    db /= m
    dw /= m

    return (J.item(), dw.tolist(), db.item())

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
n = len(weights)

l = gradient_descent_step(weights, bias, m, n, input_data, output_data)
print(l)