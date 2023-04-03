import numpy as np  #导入numpy库
import cupy as cp   #导入cupy库
import matplotlib.pyplot as plt #导入matplotlib库
from scipy.optimize import minimize #导入scipy库中的minimize函数
from sklearn.datasets import fetch_openml   #导入sklearn库中的fetch_openml函数

cp.cuda.Device(0).use() #使用GPU

def sigmoid(x,theta):
    return 1/(1+cp.exp(-cp.dot(x,theta.T)))   
    #sigmoid函数

def lossfunction(x,y,theta):
    return cp.dot((sigmoid(x,theta)-y),x)  
    #lossfunction

def model_train(x,y,label,a,m,theta):   #x[60000,785] y[60000,1] thetal[10,785]
    update = cp.zeros([1,785],dtype=float)  #初始化update
    update_avg=2.0  #初始化update_avg
    j=0 #初始化j
    plt.subplot(3,4,label+1)    #绘制子图
    plt.xlim(0,100) #设置x轴范围
    plt.ylim(0,75)  #设置y轴范围
    for i in range(60000):
        if y[i]==label:
            y[i]=1
        else:
            y[i]=0
        #判断y是否为训练目标
    while update_avg >0.5 and j<10000:
        #print(x.shape,y.shape,theta.shape,type(i))
        update=cp.dot(cp.ones([785],dtype=float),lossfunction(x,y,theta))
        theta=theta-a*(1/m)*update
        avg=cp.average(update)
        #print(x.T.shape,update.shape,theta.shape)
        if avg>0:
            update_avg = avg*a
        else:
            update_avg = -avg*a
        update_avg=update_avg.get()
        if j %50==0:
            plt.scatter(int(j/50),update_avg)
            print("The train taget ",label,"at epoch:",int((j/50)+1),"the average update is:",update_avg)
        #print("epoch",update_avg,cp.average(theta))
        i==0
        j=j+1

    return theta
    #模型训练函数

mnist=fetch_openml('mnist_784') #获取mnist数据集
x,y=mnist['data'],mnist['target']   #获取x，y参数
x=x/255     #将x正则化，以避免参数一处
shuffle_index = np.random.permutation(70000)            # 随机排列一个序列，返回一个排列的序列。
x,y=x.values[shuffle_index],y.values[shuffle_index]     #x.values是将panda frame数据类型转换为numpy数组
x=np.insert(x,0,1,axis=1)                               #插入一列1于数组首
y=cp.array(y.astype(int)-48)    #将y转换为int类型
x_train,x_test,y_train,y_test=x[:60000],x[60000: ],y[:60000],y[60000: ]
thetal=np.random.randint(0,10,size = [10,785])  #初始化thetal
x_train,x_test,thetal=cp.array(x_train),cp.array(x_test),cp.array(thetal)   #将x_train,x_test,thetal转换为cupy数组
#数据处理

a=0.05  #设置学习率
m=60000 #设置样本数量
for i in range(10):
    thetal[i]=model_train(x_train,y_train,i,a,m,thetal[i])  #训练模型
plt.show()  #显示图像
#训练模型


#评估模型


#输出模型


