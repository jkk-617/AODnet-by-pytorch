from ast import mod
import numpy as np
from sympy import N

def model(x, w, b):
    return np.dot(x, w) + b

def loss(x, w, y, b):
    return model(x, w, b) - y

def cost(x, w, y, b):
    return np.sum((loss(x, w, y, b)**2)) / (x.shape[0] * 2)

def gradient(x, w, y, b):
    loss_tmp = loss(x, w, y, b)
    x_t = x.T
    w_grad = x_t @ loss_tmp / x.shape[0]
    b_grad = np.sum(loss_tmp)  / x.shape[0]
    return w_grad, b_grad

def gradient_descent(x, w, y, b, alpha):
    w_grad, b_grad = gradient(x, w, y, b)
    w = w - alpha * w_grad
    b = b - alpha * b_grad
    return w, b

def normalize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma
    return x_norm

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
# print("========   ", np.mean(x_train, axis=0))
w = np.zeros(4)
b = np.zeros(1)
alpha = 1.e-3
iterations = 1000
x_norm = normalize(x_train)
# print(x_norm)

for i in range(10000):
    w , b = gradient_descent(x_norm, w, y_train, b, alpha)
# print(w, b)
print(model(x_norm, w, b), y_train)

# for i in range(iterations):
#     w , b = gradient_descent(x_train, w, y_train, b, alpha)
# print(w, b)
# print(model(x_train, w, b), y_train)










