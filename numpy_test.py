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

def numpy_basic_operation():
    # 索引、平均值等基本运算
    A = np.arange(2, 14).reshape((3, 4))
    print('A: ')
    print(A)
    # 索引
    print("min element's order in A: ", np.argmin(A))
    print("max element's order in A: ", np.argmax(A))
    # 均值
    print("A.mean: ", A.mean())
    # cumsum累加:生成的每一项矩阵元素均是从原矩阵首项累加到对应项的元素之和。
    print("A.cumsum: ", np.cumsum(A))
    # 累差diff：前一项于后一项之差
    print("A.diff: ", np.diff(A))
    # nonzero：将所有非零元素的行于列坐标分割开，重构成两个分别关于行和列的矩阵
    print("A.nonzero: ", np.nonzero(A))
    # sort：按照行进行排序
    B = np.arange(14,2,-1).reshape((3,4))
    print('B:')
    print(B)
    print("sort B: ", np.sort(B))
    # 转置的两种方式：transpose和T
    print("B's transpose: ", B.T)
    # clip：限定矩阵中元素的上下限
    print("B's element between 5 ann 9:")
    print(np.clip(B,5,9))

def numpy_index():
    # numpy索引、迭代的方法
    A = np.arange(3,15)
    print(A)
    print("A[3]: ", A[3])
    A = A.reshape(3,4)
    print(A)
    print("A[2]: ", A[2])
    print("A[1][1]: ", A[1][1])
    print("A[1, 1]: ", A[1, 1])
    print("A[1, 1:3]: ", A[1, 1:3])
    print("print A's row: ")
    for row in A:
        print(row)
    print("print A's column: ")
    for column in A.T:
        print(column)
    # flatten：展开成一行， flat：迭代器，object
    print("A.flatten: ", A.flatten())
    print("A.flat: ")
    for item in A.flat:
        print(item)

def numpy_merge():
# array的合并
    A = np.array([1,1,1]).reshape(-1,1) #concatenate需要两个dimensions，一维数组会报错
    B = np.array([2,2,2]).reshape(-1,1)
    print("A: ")
    print(A)
    print("B: ")
    print(B)
    print("vstack:", np.vstack((A, B))) #注意vstack((双括号))
    print("hstack:", np.hstack((A, B)))
    print("A[np.newaxis,:]")
    print(A[np.newaxis,:])
    print("A[:,np.newaxis]")
    print(A[:,np.newaxis])
    print("np.concatenate((A,B,B,A), axis=0)")
    print(np.concatenate((A,B,B,A), axis=0))
    print("np.concatenate((A,B,B,A), axis=1)")
    print(np.concatenate((A,B,B,A), axis=1))

def numpy_split():
# array的分割
    A = np.arange(12).reshape((3,4))
    print(A)
    # 纵向分割
    print("split, axis = 1: ")
    print(np.split(A, 2, axis=1))
    # 横向分割
    print("split, axis = 0: ")
    print(np.split(A, 3, axis=0))
    # 不等量的分割
    print("np.array_split, axis = 1: ")
    print(np.array_split(A, 3, axis=1))

def numpy_copy():
# 测试numpy的复制
    A = np.arange(4)
    print(A)
    print("first B = A, then update A[0] = 11, finally print(B):")
    B = A
    A[0] = 11
    print(B)
    print("A: ")
    print(A)
    print("first C = A.copy(), then update A[0] = 22, finally print(C):")
    C = A.copy()
    A[0] = 22
    print("C: ")
    print(C)
    print("A: ")
    print(A)
