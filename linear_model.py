import numpy as np
import matplotlib.pyplot as plt

def compute_cost(x, y, w, d):
    return x * w + d - y

def loss(x, y, w, d):
    sum = 0
    for i in range(len(x)):
        sum = sum + compute_cost(x[i], y[i], w, d) ** 2
    return sum / (2 * len(x))

# 因为损失函数是关于每个样本的偏差和的平均值
# 所以计算梯度也要要获取每个样本，根据偏导数求出梯度
def compute_gradient(x ,y, w, d):
    grad_w , grad_d = 0, 0
    for i in range(len(x)):
        tmp = compute_cost(x[i], y[i], w, d)
        grad_w = grad_w + tmp * x[i]
        grad_d = grad_d + tmp
    grad_w = grad_w / len(x)
    grad_d = grad_d / len(x)
    return grad_w, grad_d

# 机器学习训练过程本质就是更新权重w，d
# 更新梯度本质就是根据loss求关于权重w，d的偏导，根据步长更新w，d
def gradient_descent(x ,y, w, d, rate):
    grad_w, grad_d = compute_gradient(x ,y, w, d)
    w = w - rate * grad_w
    d = d -  rate * grad_d
    return w, d


x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
w = 0
d = 0
rate = 1.0e-2

loss_history = np.zeros(10000)
for i in range(10000):
    w , d = gradient_descent(x ,y, w, d, rate)
    loss_history[i] = loss(x, y, w, d)
print(f"w: {w} d : {d}")
print(4 * w + d)
print("===============")
print(loss_history)
