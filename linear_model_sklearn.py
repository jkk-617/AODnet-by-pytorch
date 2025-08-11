import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sympy import xthreaded

def train(x, y, iter):
    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x)
    print("============ ", x_norm)
    sgdr = SGDRegressor(max_iter=iter)
    sgdr.fit(x_norm, y)
    return sgdr

x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
iter = 1000

sgdr = train(x_train, y_train, iter)
print(sgdr.intercept_, sgdr.coef_)
scaler = StandardScaler()
x_norm = scaler.fit_transform(x_train)
print(sgdr.predict(x_norm))


'''
import numpy as np                      # 导入 NumPy 库，并使用别名 np，便于后续进行矩阵运算和数值计算

from sklearn.linear_model import SGDRegressor   # 从 scikit-learn 库中导入 SGDRegressor，用于基于随机梯度下降的线性回归

from sklearn.preprocessing import StandardScaler # 导入 StandardScaler，用来将特征按列标准化（均值为 0、方差为 1）

from sympy import xthreaded             # 从 sympy 库导入 xthreaded（未在后续代码中使用，可能是多余或笔误）

def train(x, y, iter):                  # 定义 train 函数，接受输入特征 x、目标变量 y 和迭代次数 iter
    scaler = StandardScaler()           #   创建一个 StandardScaler 实例，用于标准化处理
    x_norm = scaler.fit_transform(x)    #   对 x 进行 fit（计算均值和标准差）并 transform（应用标准化），得到 x_norm
    sgdr = SGDRegressor(max_iter=iter)  #   创建一个 SGDRegressor 实例，设置最大迭代次数为 iter
    sgdr.fit(x_norm, y)                 #   在标准化后的数据 x_norm 和目标 y 上训练模型
    return sgdr                         #   返回训练好的 SGDRegressor 对象

x_train = np.array([[2104, 5, 1, 45],   # 构造训练特征矩阵 x_train，形状为 (3,4)，每行对应一个样本，每列是一个特征
                    [1416, 3, 2, 40],
                    [852,  2, 1, 35]])

y_train = np.array([460, 232, 178])      # 构造目标向量 y_train，长度为 3，分别对应上面三个样本的房价（单位：千美元）

iter = 1000                              # 设置 SGDRegressor 的最大迭代次数为 1000

sgdr = train(x_train, y_train, iter)     # 调用 train 函数，返回已训练好的模型 sgdr

print(sgdr.intercept_, sgdr.coef_)       # 打印模型的截距（intercept_）和每个特征的回归系数（coef_）

scaler = StandardScaler()                # 再创建一个 StandardScaler，用于与训练时相同的标准化流程
x_norm = scaler.fit_transform(x_train)   # 对 x_train 重新进行标准化（注意：真实场景应使用训练时保存的 scaler）

print(sgdr.predict(x_norm))              # 对标准化后的 x_norm 调用 predict，输出模型对训练集的预测值

'''











