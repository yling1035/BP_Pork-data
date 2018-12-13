# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 09:11:43 2018

@author: yl
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
"""
#以下的语句来证明矩阵与列表的区别
print(type(a))
print(a)
print(a[0][1])
print(len(a),len(a[0])) #输出行列
print(a.shape)
print(a.size)
"""
'''
##数组与矩阵的区别,当为数组时，不能使用size和shape
b=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
print(b)
print(type(b))
print(len(b),len(b[0]))
b1=np.array(b)
print(type(b1))
print(b1)
print(b1.shape)
print(b1.size)
print(b1.shape[0],b1.shape[1])
'''
"""
x0=np.load("train0.npy")
x1=np.load("train1.npy")
y0=np.array(np.zeros(900))
y1=np.array(np.ones(215))
c0=np.load("test0a.npy")
c1=np.load("test1a.npy")
print(c0.shape,c1.shape)
#x=torch.cat((x0,x1),0).type(torch.FloatTensor)
#y=torch.cat((y0,y1),0).type(torch.FloatTensor)
#plt.sactter(x.data.numpy(),y.data.numpy)
#plt.show()
"""
x0=np.load("train0.npy")
x1=np.load("train1.npy")
y0=np.array(np.zeros(119))
y1=np.array(np.ones(215))

xtrain=np.vstack((x0,x1))
ytrain=np.hstack((y0,y1))
print(xtrain.shape)
#np.savetxt('train_data.csv',xtrain, delimiter = ',')
#np.savetxt('train_label.csv',ytrain, delimiter = ',')
#x_train=np.hstack((xtrain,ytrain))  #这行出错
  
x0_test=np.load("test0a.npy")   
x1_test=np.load("test1a.npy")
y0_test=np.array(np.zeros(20))
y1_test=np.array(np.ones(20))
   
xtest=np.vstack((x0_test,x1_test))
ytest=np.hstack((y0_test,y1_test))
#x_test=np.hstack((xtest,ytest))  #这行也出错
   
np.savetxt('test_data.csv',xtest, delimiter = ',')
np.savetxt('test_label.csv',ytest, delimiter = ',')

features=[]
for i in range(10000,10002):
    features.append(i)
    print(i)
#classlist = [featvec[-1] for featvec in train]