import numpy as np  #����numpy��
import cupy as cp   #����cupy��
import matplotlib.pyplot as plt #����matplotlib��
from scipy.optimize import minimize #����scipy���е�minimize����
from sklearn.datasets import fetch_openml   #����sklearn���е�fetch_openml����

cp.cuda.Device(0).use() #ʹ��GPU

def sigmoid(x,theta):
    return 1/(1+cp.exp(-cp.dot(x,theta.T)))   
    #sigmoid����

def lossfunction(x,y,theta):
    return cp.dot((sigmoid(x,theta)-y),x)  
    #lossfunction

def model_train(x,y,label,a,m,theta):   #x[60000,785] y[60000,1] thetal[10,785]
    update = cp.zeros([1,785],dtype=float)  #��ʼ��update
    update_avg=2.0  #��ʼ��update_avg
    j=0 #��ʼ��j
    plt.subplot(3,4,label+1)    #������ͼ
    plt.xlim(0,100) #����x�᷶Χ
    plt.ylim(0,75)  #����y�᷶Χ
    for i in range(60000):
        if y[i]==label:
            y[i]=1
        else:
            y[i]=0
        #�ж�y�Ƿ�Ϊѵ��Ŀ��
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
    #ģ��ѵ������

mnist=fetch_openml('mnist_784') #��ȡmnist���ݼ�
x,y=mnist['data'],mnist['target']   #��ȡx��y����
x=x/255     #��x���򻯣��Ա������һ��
shuffle_index = np.random.permutation(70000)            # �������һ�����У�����һ�����е����С�
x,y=x.values[shuffle_index],y.values[shuffle_index]     #x.values�ǽ�panda frame��������ת��Ϊnumpy����
x=np.insert(x,0,1,axis=1)                               #����һ��1��������
y=cp.array(y.astype(int)-48)    #��yת��Ϊint����
x_train,x_test,y_train,y_test=x[:60000],x[60000: ],y[:60000],y[60000: ]
thetal=np.random.randint(0,10,size = [10,785])  #��ʼ��thetal
x_train,x_test,thetal=cp.array(x_train),cp.array(x_test),cp.array(thetal)   #��x_train,x_test,thetalת��Ϊcupy����
#���ݴ���

a=0.05  #����ѧϰ��
m=60000 #������������
for i in range(10):
    thetal[i]=model_train(x_train,y_train,i,a,m,thetal[i])  #ѵ��ģ��
plt.show()  #��ʾͼ��
#ѵ��ģ��


#����ģ��


#���ģ��


