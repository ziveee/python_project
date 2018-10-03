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

