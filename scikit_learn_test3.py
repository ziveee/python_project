# sklearn的交叉验证方法
from __future__ import print_function
from sklearn.datasets import load_iris # iris数据集
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier # K最近领分类算法
from sklearn.model_selection import cross_val_score #K折交叉验证模块
import matplotlib.pyplot as plt # 可视化模块
import numpy as np

def basic_validation_test():
    # 基础验证法:
    # 加载数据:
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 分割数据集：
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    # 建立模型:
    knn = KNeighborsClassifier()
    # 训练模型：
    knn.fit(X_train, y_train)
    # 打印准确率
    print("X_test: \n", X_test)
    print("y_test: \n", y_test)
    print("knn score: \n", knn.score(X_test, y_test))
    return 0

def cross_validation_test():
    # 交叉验证法：
    # 加载数据:
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 分割数据集：
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    # 建立模型:
    knn = KNeighborsClassifier()
    # 训练模型：
    knn.fit(X_train, y_train)
    # 使用k折交叉验证模块：
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    # 打印5次的预测准确率：
    print("5 times prediction result:\n", scores)
    print("scores.mean:\n", scores.mean())
    return 0

def based_on_accuracy():
    # 加载数据:
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 分割数据集：
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    # 基于准确率来判断
    # 建立测试集：
    k_range = range(1, 31)
    k_scores = []
    # 通过迭代的方法来计算不同参数对模型的影响，并返回交叉验证后的平均验证率
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    # 可视化数据
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-validated Accuracy')
    print("The best K is: \n", np.argmax(k_scores))
    plt.show()
    return 0

def based_on_mean_squared_error():
    # 基于平均方差来判断回归模型的好坏
    # 加载数据:
    iris = load_iris()
    X = iris.data
    y = iris.target
    # 分割数据集：
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
    # 基于平均方差来判断:
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        lose = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
        k_scores.append(lose.mean())
    # 可视化数据
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated MSE')
    print("The best K is:\n", np.argmin(k_scores))
    plt.show()
    return 0

def cross_validation_test_list():
    # sklearn的交叉验证方法测试
    # 基础验证法：
    # basic_validation_test()
    # 交叉验证法：
    # cross_validation_test()
    # 基于准确率来判断：
    # based_on_accuracy()
    # 基于平均方差来判断：
    based_on_mean_squared_error()
    return 0
