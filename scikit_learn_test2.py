from __future__ import print_function
# 标准化数据模块
from sklearn import preprocessing
import numpy as np
# 分割数据模块
from sklearn.model_selection import train_test_split
# 生成适合classification数据的模块
from sklearn.datasets.samples_generator import make_classification
# 支持向量机分类模块
from sklearn.svm import SVC
# 可视化数据模块
import matplotlib.pyplot as plt

def Normalization_test():
    # 生成具有两种特征的300个数据
    X, y = make_classification(n_samples=300,
                               n_features=2,
                               n_redundant=0,
                               n_informative=2,
                               random_state=22,
                               n_clusters_per_class=1,
                               scale=100)
    
    # 可视化数据
    # plt.scatter(X[:,0], X[:,1], c=y)
    # plt.show()
    # 使用SVC来预测
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf1 = SVC()
    clf1.fit(X_train, y_train)
    print("before normalization, svc's score:")
    print(clf1.score(X_test, y_test))
    # 进行数据标准化
    X = preprocessing.scale(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf1 = SVC()
    clf1.fit(X_train, y_train)
    print("after normalization, svc's score:")
    print(clf1.score(X_test, y_test))
    return 0
