import matplotlib.pyplot as plt
import numpy as np

def basic_operation():
    # 基础应用示例
    x = np.linspace(-1, 1, 50) # linspace：-1到1之间定义50个数
    y = 2*x + 1
    print(x, y)
    plt.figure()
    plt.plot(x, y)
    plt.show()

def simple_line_test():
    # 简单线条
    x = np.linspace(-3, 3, 50)
    y1 = 2*x + 1
    y2 = x**2
    plt.figure(num=3, figsize=(8,5),)
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.show()

def set_axis_test():
    # 设置坐标轴
    x = np.linspace(-3, 3, 50)
    y1 = 2*x + 1
    y2 = x**2
    plt.figure()
    plt.plot(x, y2)
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
    plt.xlim((-1, 2)) # 设置x轴
    plt.ylim((-2, 3)) # 设置y轴
    plt.xlabel('I am x')
    plt.ylabel('I am y')
    # plt.show()
    
    # 调整名字和间隔
    new_ticks = np.linspace(-1, 2, 5)
    print(new_ticks)
    plt.xticks(new_ticks)
    plt.yticks([-2,-1.8,-1,1.22,3],[r'$relly\ bad$',r'$bad$',r'$normal$',r'$good$',r'$relly\ good$'])
    plt.show()

def move_axis_test():
    # 移动坐标轴
    x = np.linspace(-3,3,50)    #在-3与3之间取50个点
    y1 = 2*x + 1    #定义y1
    y2 = x**2   #定义y2
    plt.figure()    #定义图像窗口
    plt.plot(x, y2) #画出y2曲线
    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--') #画出y1曲线，红色，线宽1.0，类型为虚线
    plt.xlim((-1,2))    # 画出x轴
    plt.ylim((-2,3))    # 画出y轴
    new_ticks = np.linspace(-1,2,5) #在-1和2之间取5个数
    plt.xticks(new_ticks)   #设置x轴刻度
    plt.yticks([-2,-1.8,-1,1.22,3],['$relly\ bad$','$bad$','$normal$','$good$','$relly\ good$'])    #设置y轴刻度
    ax = plt.gca()  #获取当前坐标轴信息
    ax.spines['right'].set_color('none')    #设置右侧边框颜色
    ax.spines['top'].set_color('none')  #设置上侧边框颜色
    ax.xaxis.set_ticks_position('bottom')   # 调整x坐标轴上刻度数字或名称的位置
    ax.spines['bottom'].set_position(('data', 0))   # 设置x轴的位置为y=0
    ax.yaxis.set_ticks_position('left') # 调整y坐标轴上刻度数字或名称的位置
    ax.spines['left'].set_position(('data', 0)) # 设置y轴的位置为x=0
    plt.show()

def add_label_test():
    # 添加标签
    x = np.linspace(-3,3,50)    #在-3与3之间取50个点
    y1 = 2*x + 1    #定义y1
    y2 = x**2   #定义y2
    plt.figure()    #定义图像窗口
    plt.xlim((-1,2))    # 画出x轴
    plt.ylim((-2,3))    # 画出y轴
    new_ticks = np.linspace(-1,2,5) #在-1和2之间取5个数
    plt.xticks(new_ticks)   #设置x轴刻度
    plt.yticks([-2,-1.8,-1,1.22,3],['$relly\ bad$','$bad$','$normal$','$good$','$relly\ good$'])    #设置y轴刻度
    # 画出y1,y2,设置label
    l1, = plt.plot(x, y1, label='linear line')
    l2, = plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--', label='square line')
    plt.legend(loc='upper right')
    # 重新设置label
    plt.legend(handles=[l1, l2], labels=['l1','l2'], loc='best') #l1，l2参数以,结尾，因为plot返回的是一个列表，loc=‘best’表示自动分配最佳位置
    plt.show()

def annotation_test():
    # annotation标注
    # 绘制一条直线
    x = np.linspace(-3,3,50)
    y = 2*x + 1
    plt.figure(num=1, figsize=(8, 5),)
    plt.plot(x, y, )
    # 移动坐标轴
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    # 画一条垂直于x轴的虚线
    x0 = 1
    y0 = 2*x0 + 1
    plt.plot([x0, x0,], [0,y0,], 'k--', linewidth=2.5) # 画线
    plt.scatter([x0,], [y0,], s=50, color='b') # 描点
    # 标注（x0，y0）的信息
    plt.annotate(r'$2x+1=%s$' % y0, xy=(x0,y0), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    # 添加注释text
    plt.text(-3.7, 3, r'$This\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
             fontdict={'size':16, 'color':'r'})
    plt.show()

def plt_scatter_test():
    # 散点图
    n = 1024 #data size
    X = np.random.normal(0,1,n) #每个点的X值
    Y = np.random.normal(0,1,n) #每个点的y值
    T = np.arctan2(Y,X) # 每个点的颜色
    plt.scatter(X, Y, s=75, c=T, alpha=.5) #size=75，透明度50%
    plt.xlim(-1.5, 1,5) #X轴的范围
    plt.xticks(()) #隐藏x轴
    plt.ylim(-1.5, 1.5)
    plt.yticks(())
    plt.show()

def plt_bar_test():
    # 柱状图
    n = 12
    X = np.arange(n)
    print(X)
    # 生成基本图形
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    plt.bar(X, +Y1)
    plt.bar(X, -Y2)
    plt.xlim(-.5, n)
    plt.xticks(())
    plt.ylim(-1.25, 1.25)
    plt.yticks(())
    # 添加颜色
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    # 添加数据
    for x,y in zip(X, Y1):
        print(x,y)
        plt.text(x, y+0.05, '%.2f' % y, ha='center', va='bottom')
    for x,y in zip(X, Y2):
        print(x,y)
        plt.text(x, -y-0.05, '%.2f' % y, ha='center', va='top')
    plt.show()

def plt_contourf_test():
    # 等高线图
    def f(x,y):
        return (1 - x / 2 + x**5 + y**3) * np.exp(-x**2 - y**2)
    n = 256
    x = np.linspace(-3, 3, n)
    y = np.linspace(-3, 3 ,n)
    X,Y = np.meshgrid(x, y)
    # 填充颜色
    plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap=plt.cm.hot)
    # 绘制等高线
    C = plt.contour(X, Y, f(X,Y), 8, colors='black')
    # 添加高度数字
    plt.clabel(C, inline=True, fontsize=10)
    # 隐藏坐标轴
    plt.xticks(())
    plt.yticks(())
    plt.show()

def matplotlib_test_list():
    # 基础应用实例
    basic_operation()
    # 简单线条
    # simple_line_test()
    # 设置坐标轴
    # set_axis_test()
    # 移动坐标轴
    # move_axis_test()
    # 添加标签
    # add_label_test()
    # Annotation标注
    # annotation_test()
    # 散点图
    # plt_scatter_test()
    # 柱状图
    # plt_bar_test()
    # 等高线图
    # plt_contourf_test()
