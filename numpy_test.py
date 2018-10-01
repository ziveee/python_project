# practice numpy

import numpy as np

def numpy_attributes():
    # Numpy属性
    array = np.array([[1,2,3],[2,3,4]])
    print(array)
    print('number of dim: ', array.ndim)
    print('shape: ', array.shape)
    print('size: ', array.size)

def numpy_array():
    # 创建数组
    a = np.array([2,23,4])
    print(a)

    # 指定数据dtype
    b = np.array([2,23,4], dtype=np.float)
    print(b, 'dtype: ', b.dtype)

    # 创建特定数据
    c = np.zeros((3,4))
    print('zeros34: ', c)
    d = np.ones((3,4), dtype=np.int)
    print('ones34: ', d)
    e = np.empty((3,4))
    print('empty34: ', e)

    # 创建连续数组：
    f = np.arange(10, 20, 2) # 10-19的数据，2步长
    print('10-19, 2:', f)

    # reshape改变数据形状
    g = np.arange(12).reshape((3,4)) # 3行4列，0到11
    print(g)

    # linspace创建线段型数据
    h = np.linspace(1, 10, 20) # 开始端1，结束端10，分割成20个数据，生成线段
    print(h)
    h = h.reshape((5,4)) # 同样可以reshape
    print(h)

def numpy_operation1():
    # 一维数组运算
    a = np.array([10,20,30,40])
    b = np.arange(4)
    print('a = ',a,' b = ',b)
    print('a+b = ', a+b)
    print('a-b = ', a-b)
    print('a*b = ', a*b)
    print('b**2 = ', b**2)
    print('10*sin a = ', 10*np.sin(a))
    print('element in b < 3:',b<3)

def numpy_operation2():
    # 多维数组运算
    a = np.array([[1,1],[0,1]])
    b = np.arange(4).reshape((2,2))
    print('a = ',a,' b = ',b)
    print('a_dot_b = ', np.dot(a, b))
    c = np.random.random(((2,4)))
    print('c = ', c)
    print('c_sum = ', np.sum(c))
    print('c_max = ', np.max(c))
    print('c_min = ', np.min(c))
    print('c_rows_sum = ', np.sum(c, axis=1))
    print('c_rows_max = ', np.max(c, axis=1))
    print('c_columns_sum = ', np.sum(c, axis=0))
    print('c_columns_min = ', np.min(c, axis=0))
