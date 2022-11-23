# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:49:00 2022


Practice of general machine learning algorithms

@author: Yuanyuan_Tang
"""



# https://www.geeksforgeeks.org/least-square-regression-line/
'''
y=ax+b

[a,b]=Y * [X, 1]* inv([X, 1]^T*[X, 1])
'''

'''
Initializing parameters

'''

import numpy as np


X = [95, 85, 80, 70, 60 ]
Y = [90, 80, 70, 65, 60 ]


m=len(X)
U_v=[1]*m
X1=[]

X1.append(X)
X1.append(U_v)

a=np.dot(Y,np.dot(np.transpose(X1),np.linalg.inv(np.dot(X1,np.transpose(X1)))))
print(a)