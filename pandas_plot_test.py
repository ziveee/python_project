import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

def series_plot_test():
    # 随机生成1000个数据
    data = pd.Series(np.random.randn(1000), index=np.arange(1000))
    # 累加该数据
    data1 = data.cumsum()
    print(data1.describe())
    # pandas，数据可以直接观看其可视化形式
    data1.plot()
    plt.show()

def dataframe_plot_test():
    data = pd.DataFrame(np.random.randn(1000,4),
                        index=np.arange(1000),
                        columns=list("ABCD"))
    data1 = data.cumsum()
    print(data1.describe())
    data1.plot()
    plt.show()

def datafram_plot_scatter():
    data = pd.DataFrame(np.random.randn(1000,4),
                        index=np.arange(1000),
                        columns=list("ABCD"))
    print(data.describe())
    ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class1')
    data.plot.scatter(x='A', y='C', color='LightGreen', label='Class2', ax=ax)
    plt.show()


def pandas_plot_test_list():
    # 测试series的可视化
    # series_plot_test()

    # 测试DataFrame的可视化
    # dataframe_plot_test()

    # 测试DataFramescatter
    datafram_plot_scatter()

