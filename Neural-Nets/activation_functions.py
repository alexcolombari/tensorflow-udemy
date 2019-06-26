import numpy as np
#import tensorflow as tf

def stepFunction(x):
    if(x >= 1):
        return 1
    else:
        return 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanhFunction(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    if(x >= 0):
        return x
    return 0

def linearFunction(x):
    return x

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

x1 = sigmoid(2.1)
print(x1)
x2 = tanhFunction(2.1)
print(x2)
x3 = relu(2.1)
print(x3)
x4 = linearFunction(2.1)
print(x4)