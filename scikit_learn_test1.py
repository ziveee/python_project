from __future__ import print_function
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def KNN_classifier_test():
    # KNeighborsClassifier Test
    # 创建数据
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    print(iris_X[:2,:])
    print(iris_y)
    # 把数据集分为训练集和测试集(30%)并乱序
    X_train, X_test, y_train, y_test = train_test_split(
        iris_X, iris_y, test_size=0.3)
    print(y_train)
    # 建立模型
    knn = KNeighborsClassifier()
    # 训练
    knn.fit(X_train, y_train)
    # 预测
    print("predict y_test:")
    print(knn.predict(X_test))
    print("compare y_test:")
    print(y_test)
    return 0

def sklearn_dateset_test():
    # 测试sklearn的数据库
    # 导入数据
    loaded_data = datasets.load_boston()
    data_X = loaded_data.data
    data_y = loaded_data.target
    # 定义模型（默认参数），后续可通过预测的准确度，进行调整，尝试不同的model和参数
    model = LinearRegression()
    # 训练模型
    model.fit(data_X, data_y)
    # 打印X的前4个预测值
    print("predict target:")
    print(model.predict(data_X[:4,:]))
    print("actual data:")
    print(data_y[:4])
    
    # LinearRegressor model的参数：斜率和截距
    print("model.coef_:\n", model.coef_)
    print("model.intercept_:\n", model.intercept_)
    # 取出之前定义的参数
    print("model.get_params:\n", model.get_params())
    # model.score(data_X, data_y)可以对Model以R^2的方式进行打分，输出精确度
    # R2越大（接近于1），所拟合的回归方程越优
    print("model.score:\n", model.score(data_X, data_y))

    # 创建虚拟数据，100个样本，1个特征，1个目标，noise越大，点越离散
    X1, y1 = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
    X2, y2 = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=50)
    plt.scatter(X1, y1)
    plt.scatter(X2, y2)
    plt.show()
    return 0

def sklearn_test_list():
    # 测试sklearn相关算法
    
    # KNN_classifier_test()
    # 测试KNN算法
    
    # 测试sklearn的数据库
    sklearn_dateset_test()
    return 0
