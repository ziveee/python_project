import pandas as pd
import numpy as np

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

def pandas_definition():
# 测试pandas的基本概念
    # Series:
    s = pd.Series([1,3,6,np.nan,44,1])
    print("Series: ")
    print(s)
    # DataFrame:
    dates = pd.date_range('20180101', periods=6)
    df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=['a','b','c','d'])
    print("DataFrame: ")
    print(df)
    print(df['b'])

def pandas_simple_operation():
# 测试pandas的简单运用
    # 创建不指定行标签和列标签的dataframe
    df1 = pd.DataFrame(np.arange(12).reshape((3,4)))
    print(df1)
    # 创造每一列数据
    df2 = pd.DataFrame({'A' : 1.,
                        'B' : pd.Timestamp('20181001'),
                        'C' : pd.Series(1, index=list(range(4)),dtype='float32'),
                        'D' : np.array([3]*4, dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo'})
    print(df2)
    # 查看每一列中数据的类型：
    print("df2's dtypes: \n ", df2.dtypes)
    # 查看列的序列号：
    print("df2.index: \n", df2.index)
    # 查看列的名称：
    print("df2.columns: \n", df2.columns)
    # 查看df2的所有值：
    print("df2.values: \n", df2.values)
    # 查看df2的描述
    print("df2.describe():\n",df2.describe())
    # 翻转df2
    print("df2.T: \n", df2.T)
    # sort_index
    print("df2.sort_index: \n", df2.sort_index(axis=1, ascending=False))
    # sort_values(by='B')
    print("df2.sort_values: \n", df2.sort_values(by='B'))

def pandas_choose_data():
#   测试pandas的选择数据
    dates = pd.date_range('20180101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A','B','C','D'])
    print("df: \n", df)
    print("df[0:3] \n", df[0:3])
    print("df['20180101':'20180104'] \n", df['20180101':'20180104'])
    print("df.loc['20180102'] \n", df.loc['20180102'])
    print("df.loc[:,['A','B']] \n", df.loc[:,['A','B']])
    print("df.iloc[3,1] \n", df.iloc[3,1])
    print("df.iloc[3:5,1:3] \n", df.iloc[3:5,1:3])
    print("df.iloc[[1,3,5],1:3]) \n", df.iloc[[1,3,5],1:3])
    print("df.ix[:3, ['A','C']] \n", df.ix[:3, ['A','C']])
    print("df[df.A > 8] \n", df[df.A > 8])

def pandas_NaNData_test():
# pandas处理丢失数据

    # 建立一个6*4的矩阵，把其中两个位置置空
    dates = pd.date_range('20180101', periods=6)
    df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A','B','C','D'])
    df.iloc[0,1] = np.nan
    df.iloc[1,2] = np.nan
    print("df: \n", df)

    print("\n drop all rows or columns include NaN: ")
    df1 = df.dropna(
              axis=0, #0: 对行进行操作；1: 对列进行操作
              how='any' #‘any’：只要存在NaN就drop； ‘all’：只有全是NaN才drop
              )
    print(df1)
    print("\n df.fillna(value=0) ")
    df2 = df.fillna(value=0)
    print(df2)
    print("\n df.isnull()")
    print(df.isnull())

def pandas_concat_test():
# 测试pandas concat函数
    # 定义资料集
    df1 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'])
    print(df1)
    df2 = pd.DataFrame(np.ones((3,4))*1, columns=['a','b','c','d'])
    print(df2)
    df3 = pd.DataFrame(np.ones((3,4))*2, columns=['a','b','c','d'])
    print(df3)
    # 纵向合并
    print("pd.concat([df1, df2, df3], axis=0)")
    print(pd.concat([df1, df2, df3], axis=0))
    print("ignore_index:True")
    print(pd.concat([df1, df2 ,df3], axis=0, ignore_index=True))
    # join方式合并
    df4 = pd.DataFrame(np.ones((3,4))*0, columns=['a','b','c','d'], index=[1,2,3])
    print(df4)
    df5 = pd.DataFrame(np.ones((3,4))*1, columns=['b','c','d','e'], index=[2,3,4])
    print(df5)
    print("join='outer'")
    print(pd.concat([df4, df5], axis=0, join='outer', sort=True))
    print("join='inner' and ignore_index=True")
    print(pd.concat([df4, df5], axis=0, join='inner', sort=True, ignore_index=True))
    # append方式纵向合并
    print("df1, df2, df3, s1: ")
    print(df1)
    print(df2)
    print(df3)
    s1 = pd.Series([1,2,3,4], index=['a','b','c','d'])
    print(s1)
    print("df1.append(s1, ignore_index=True)")
    print(df1.append(s1, ignore_index=True))
    print("df1.append(df2) and ignore_index=True:")
    print(df1.append(df2, ignore_index=True))
    print("df1.append([df2,df3]......)")
    print(df1.append([df2, df3], ignore_index=True))
    
    # 横向合并
    print("df4 and df5: \n", df4, "\n", df5)
    print("axis=1")
    print(pd.concat([df4, df5], axis=1))
    print("axis=1, join_axes=[df4.index]")
    print(pd.concat([df4, df5], axis=1, join_axes=[df4.index]))

def pandas_merge_test():
# 测试pandas的merge函数
    left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                        'A':['A0','A1','A2','A3'],
                        'B':['B0','B1','B2','B3']
                        })
    right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                         'C':['C0','C1','C2','C3'],
                         'D':['D0','D1','D2','D3']
                         })
    print(left)
    print(right)
    print("meige left and right:")
    print(pd.merge(left, right, on='key'))
    # 依据key1与key2 columns进行合并，并打印出四种结果['left', 'right', 'outer', 'inner']
    left1 = pd.DataFrame({'key1':['k0','k0','k1','k2'],
                         'key2':['k0','k1','k0','k1'],
                         'A':['a0','a1','a2','a3'],
                         'B':['b0','b1','b2','b3']
                         })
    right1 = pd.DataFrame({'key1':['k0','k1','k1','k2'],
                          'key2':['k0','k0','k0','k0'],
                          'C':['c0','c1','c2','c3'],
                          'D':['d0','d1','d2','d3']
                         })
    print(left1)
    print(right1)
    print("inner merge:")
    print(pd.merge(left1, right1, on=['key1','key2'], how='inner'))
    print("outer merge:")
    print(pd.merge(left1, right1, on=['key1','key2'], how='outer'))
    print("left merge:")
    print(pd.merge(left1, right1, on=['key1','key2'], how='left'))
    print("right merge:")
    print(pd.merge(left1, right1, on=['key1','key2'], how='right'))
    
    # indicator参数
    df1 = pd.DataFrame({'col1':[0,1], 'col_left':['a','b']})
    df2 = pd.DataFrame({'col1':[1,2,2], 'col_right':[2,2,2]})
    print("df1 and df2:")
    print(df1)
    print(df2)
    print("indicator = True:")
    print(pd.merge(df1, df2, on='col1', how='outer', indicator=True))
    print("indicator = indicator_col:")
    print(pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_col'))

    # 依据index来进行合并
    left2 = pd.DataFrame({'A':['a0','a1','a2'],
                         'B':['b0','b1','b2']},
                         index=[0,1,2])
    right2 = pd.DataFrame({'C':['c0','c1','c2'],
                          'D':['d0','d1','d2']},
                         index=[0,2,3])
    print("left2 and right2:")
    print(left2)
    print(right2)
    print("left_index and right_index = True, how=outer: ")
    print(pd.merge(left2, right2, left_index=True, right_index=True, how='outer'))
    print("how=inner:")
    print(pd.merge(left2, right2, left_index=True, right_index=True, how='inner'))

    # overlapping问题
    boys = pd.DataFrame({'k':['k0','k1','k2'], 'age':[1,2,3]})
    girls = pd.DataFrame({'k':['k0','k0','k3'], 'age':[4,5,6]})
    print("boys and girls:")
    print(boys)
    print(girls)
    print("inner merge, suffixes=['_boy','_girl']")
    print(pd.merge(boys, girls, on='k', suffixes=['_boy','_girl'], how='inner'))
                              
def pandas_test_list():
    # 测试pandas的基本概念
    pandas_definition()

    # 测试pandas的简单运用
    # pandas_simple_operation()

    # 测试pandas的选择数据
    # pandas_choose_data()

    # pandas处理丢失数据
    # pandas_NaNData_test()

    # 测试pandas concat函数
    # pandas_concat_test()

    # 测试pandas merge函数
    # pandas_merge_test()
