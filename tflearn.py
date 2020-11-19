import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from matplotlib import pyplot as plt


def load_data():
    # 共150条数据，训练120条，测试30条，进行2,8分进行模型训练
    # 每条数据类型为 x{nbarray} [6.4, 3.1, 5.5, 1.8]
    inputdata = datasets.load_iris()
    # 切分，测试训练2,8分
    x_train, x_test, y_train, y_test = \
        train_test_split(inputdata.data, inputdata.target, test_size = 0.2, random_state=0)
    return x_train, x_test, y_train, y_test

# def main():
#     # 训练集x ,测试集x,训练集label,测试集label
#     x_train, x_test, y_train, y_test = load_data()
#     # l2为正则项
#     model = LogisticRegression(penalty='l2')
#     model.fit(x_train, y_train)

#     print ("w: ", model.coef_)
#     print ("b: ", model.intercept_)
#     # 准确率
#     print ("precision: ", model.score(x_test, y_test))
#     print ("MSE: ", np.mean((model.predict(x_test) - y_test) ** 2))

epoch = 1000
lr = 0.03

def sigmod(z):
    return 1.0 / (1.0 + np.exp(-z))

def loss_fun(a,y,m):
    print(a)
    print(y)
    print(m)
    exit()
    return -(np.sum(y * np.log(a) + (1-y) * np.log(1-a))) * (1.0/m)

def error(y_pre,y_true):
    # y_pre[np.where(y_pre>0.5)] = 1
    np.where(y_pre>0.5,1,0)
    error = np.mean(y_pre == y_true)
    
    return error

def norm(x):
    x_max = np.max(x)
    x_min = np.min(x)

    x = (x-min)/(x_max - x)
    return x 

def show(x0,x1,y):
    plt = figure()
    plt.subplot(111)
    plt.scatter(x0[:,0],x0[:,-1],color='r')
    plt.scatter(x1[:,0],x1[:,-1],color='b')
    plt.plot(range(0,y.shape[0]),y,color='g')

    plt.show()


def main(x_train,y_train):

    w = np.random.randn(x_train.shape[0],1)
    b = np.random.randn(1)
    m = x_train.shape[1]
    l = []
    for i in range(epoch):
        z = np.dot(w.T,x_train) + b
        a = sigmod(z)
        
        loss = loss_fun(a,y_train,m)
        l.append(loss)
        print('loss',loss)

        err = error(a.copy(),y_train)
        print('error',err)

        dz = a - y_train
        dw = np.dot(x_train, dz.T) * 1/m
        db = sum(dz) * (1/m) 

        w = w - lr*dw
        b = b - lr*db

    return w, b,np.array(l)


    


if __name__ == '__main__':

    x_train,x_test,y_train,y_test = load_data()
    w,b,loss = main(x_train.T,y_train)

    # plt.plot(range(0,loss.shop[0]),loss)
    # plt.show()

    print(w,999999)
    print(b)
