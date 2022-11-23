# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:12:09 2022

@author: Yuanyuan_Tang

Import dataset from Pytorch, and perform SVM and Gradient descendant

"""


import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats as st
import math

from numpy.linalg import norm
from numpy.linalg import eig



'''
### 2.1 Load training data and plot it

Note how the data is loaded with `train = True`. 
You can load the test data in a similar fashion, just set `train = False`. 
Feel free to reuse my plot function!
In this section, given two labels, we can pick two types of images.
  For simplicity, we define one type as negative images and another type as positive images. 
'''


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 10000  # Batch size is 10000
train_data = FashionMNIST("./data", train = True, download = True,
                          transform=transforms.ToTensor())  # Load training data
print(np.shape(train_data))
data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = False) # 

width = 28
height = 28
input_size = width * height

def plot_images(batch, rows, cols, title = ""):
    plt.figure(figsize = (rows, cols))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(
        vutils.make_grid(batch[:(rows * cols)], nrow = rows, normalize = True).cpu(),
        (1, 2, 0)))

first_batch = next(iter(data_loader))  # Present data one by one
plot_images(first_batch[0], 10, 10, "Some Training Data")


#print(np.shape(first_batch[0])) # The training data
#print(np.shape(first_batch[1]))   # The labels

'''
Function: Acquire positive or negative labels
'''
## Some functions in this range
def labels_twotypes(first_batch, Index_poslabel, Index_neglabel):
    N_index_label=6    # The index_label search range in case that two indexes above have the same label
    Pos_label=first_batch[1][Index_poslabel]   # The label of positive images

    for i in range(N_index_label):
        if first_batch[1][Index_poslabel]!=first_batch[1][Index_neglabel]:
            Neg_label=first_batch[1][Index_neglabel] # Check whether nagative label are the same of positive
            break;
        Index_neglabel=Index_neglabel+1
    
    return Pos_label, Neg_label

'''
# Function: list of pictures to tensors: transfer lists of images to tensors
'''
def List2Tensor(Img_list):
    N_list=len(Img_list)
    Img_tensor=torch.zeros(N_list,1, width, height)
    for i in range(N_list):
        Img_tensor[i,:,:,:]=Img_list[i]
    return Img_tensor

'''
## Pick two types of images
'''
def pick_twotype_imgs(first_batch,Pos_label, Neg_label, batch_size):
    Img_neg=[]
    Img_pos=[]
    Img_train=[]
    label_train=[]
    for i in range(batch_size):
        if first_batch[1][i]==Pos_label:
            Img_pos.append(first_batch[0][i])   # Obtain the positive images    
            label_train.append(1)
            Img_train.append(first_batch[0][i])   # Obtain the positive images    
        if first_batch[1][i]==Neg_label:
            Img_neg.append(first_batch[0][i])    # Obtain the negative images
            label_train.append(0)
            Img_train.append(first_batch[0][i])   # Obtain the positive images    
            
    return Img_pos, Img_neg, Img_train, label_train


'''
## Pick two types of pictures, pictures with lable i,j are negative and positive, respectively;
'''
# The label of positive and negative images
Index_poslabel=0
Index_neglabel=1

# Choose two labels as positive and negative images
Pos_label, Neg_label=labels_twotypes(first_batch, Index_poslabel, Index_neglabel)


# Show the two labels of images
print('Positive label:',Pos_label)
print('Negative label:',Neg_label)


# Pick the negative and positive pictures
Img_pos, Img_neg, Img_train, label_train=pick_twotype_imgs(first_batch, Pos_label, Neg_label, batch_size)

# Transfer image_list to tensor, and show negative and positive images
Img_pos=List2Tensor(Img_pos)
Img_neg=List2Tensor(Img_neg)
print(Img_pos.size())
print(Img_neg.size())
#print(len(Img_train))
#print('labels_train=:', sum(label_train))

plot_images(Img_pos, 10, 10, "Some positive pictures")
plot_images(Img_neg, 10, 10, "Some negative pictures")


'''
The other lists of functions
'''

# Function: list of pictures to tensors
def List2TrainMatrix(Img_list):
    N_list=len(Img_list)
    Img_matrix=torch.zeros(N_list,input_size+1)
    for i in range(N_list):
        Img_matrix[i,0:input_size]=Img_list[i].view(1,input_size)
    Img_matrix[:,input_size]=1
    
    return Img_matrix


## Function: obtain the stepsize
def stepsize_train(x):
    var_x=np.matmul(x,np.transpose(x))
    value_eig,vec_eig=eig(var_x/4)    
    #print(value_eig)
    beta=max(value_eig)
    
    return 1/beta   




# Logistic function to compute the probability
def logistic_function(theta, x):
    a=np.inner(theta,x)
    p=1/(1+np.exp(-a))
    
    return p


# The gradient of data 
def gradient_logistic(x,label, theta):
    N_sample=len(label)
    theta_new=0*theta   # The new gradient
    for i in range(N_sample):
        P_theta=logistic_function(theta, x[i,:])
        theta_new=theta_new+(P_theta-label[i])*x[i,:]
    
    return theta_new



# The cost function
def Loglikelihood_loss(x, label,theta):
    Loss_log=0
    N_sample=len(label)
    for i in range(N_sample):
        P_x=logistic_function(theta, x[i,:])
       # Loss_log=Loss_log+(label[i]*math.log(P_x)+(1-label[i])*math.log(1-P_x))
        if P_x==1:
            Loss_log=Loss_log-label[i]*math.log(P_x)
        elif P_x==0:
            Loss_log=Loss_log-(1-label[i])*math.log(1-P_x)
        else:
            Loss_log=Loss_log-(label[i]*math.log(P_x)+(1-label[i])*math.log(1-P_x))
        #Loss_log=Loss_log+pow((P_x-label[i]),2)
    
    return Loss_log
    
    

# The Gradient Descendant method
def Grad_Desc_Method(x, N_itn, step_size, label, theta):
    dim_theta=input_size
    #print('Theta=:',np.shape(theta))
    Loss_log_list=np.zeros(N_itn)
    theta_list=[]
    for j in range(N_itn):
        Loss_log=Loglikelihood_loss(x, label,theta)
        Loss_log_list[j]=Loss_log
        theta_new=gradient_logistic(x, label, theta)
        theta=theta-step_size*theta_new
        theta_list.append(theta)
       # if (j+1)%10==0:
        #    print('The '+format(j+1)+'-th iteration')

    
    return theta_list, Loss_log_list
    

'''
Simple training example
'''
## The gradient method
#Initialize the parameter
step_size=1/20000
N_itn=30
theta=np.random.rand(1, input_size+1)
theta=theta/10


# Obtain the training data
Img_train_GD=List2TrainMatrix(Img_train)

# Obtain the train data
#step_size=stepsize_train(Img_train_GD)

#print(torch.sum(Img_train[0].view(1,input_size)-Img_train_GD[0,0:input_size]))
#print(Img_train_GD.size())
#print(label_train)

#Fixed-stepsize Gradient descendant  Transfer pytorch data tor numpy
theta_list, Loss_log_list=Grad_Desc_Method(Img_train_GD.numpy(), N_itn, step_size, label_train, theta)


#Show the likelihood loss function
plt.figure()
plt.plot(Loss_log_list,'r-*')
plt.title('The loss of fixed-stepsize Gradient Descent method')
plt.xlabel('Iterative times')
plt.ylabel('Log-likelihood loss')


'''
Training data by  SVM algorithm
'''

# import SVC classifier
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score


#print(first_batch[1][0:10])

'''
Function: Acquire positive or negative labels
'''
## Some functions in this range
def labels_Multiletypes(first_batch, Index_poslabel, Index_neglabel):
    N_index_label=6    # The index_label search range in case that two indexes above have the same label
    Pos_label=first_batch[1][Index_poslabel]   # The label of positive images

    for i in range(N_index_label):
        if first_batch[1][Index_poslabel]!=first_batch[1][Index_neglabel]:
            Neg_label=first_batch[1][Index_neglabel] # Check whether nagative label are the same of positive
            break;
        Index_neglabel=Index_neglabel+1
    
    return Pos_label, Neg_label



'''
# Function: list of pictures to tensors
'''
def List2TrainMatrix_SVM(Img_list):
    N_list=len(Img_list)
    Img_matrix=torch.zeros(N_list,input_size)
    for i in range(N_list):
        Img_matrix[i,0:input_size]=Img_list[i].view(1,input_size)

    
    return Img_matrix



N_train=5000
x=List2TrainMatrix_SVM(first_batch[0])
#print(np.shape(label_train))
#y=label_train
y=first_batch[1]
x_train=x[0:N_train,:]
y_train=y[0:N_train]
x_test=x[N_train:,:]
y_test=y[N_train:]

#print(np.shape(x_train))



'''
---------------------------------------------------------
SVM https://www.kaggle.com/code/prashant111/svm-classifier-tutorial/notebook
'''

# rbf
print('----------------------rbf-----------------------------')
SVM=SVC()
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # linear 
# rbf
print('----------------------linear-----------------------------')
SVM=SVC(kernel="linear")
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



# # polymer

print('----------------------Polymer-----------------------------')
SVM=SVC(kernel="poly")
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # sigmod
print('----------------------sigmoid-----------------------------')
SVM=SVC(kernel="sigmoid")
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



# rbf
print('----------------------rbf,C=40-----------------------------')
SVM=SVC(C=40)
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # linear 
# rbf
print('----------------------linear,C=40-----------------------------')
SVM=SVC(kernel="linear",C=40)
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))



# # polymer

print('----------------------Polymer,C=40-----------------------------')
SVM=SVC(kernel="poly",C=40)
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# # sigmod
print('----------------------sigmoid,C=40-----------------------------')
SVM=SVC(kernel="sigmoid",C=40)
SVM.fit(x_train,y_train)

y_pred=SVM.predict(x_test)

# compute and print accuracy score
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
