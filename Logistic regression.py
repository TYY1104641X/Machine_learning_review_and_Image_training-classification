# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 23:33:28 2021

@author: Yuanyuan_Tang

Logistic regression-hands-on experience

"""




'''
Import Module
'''

import numpy as np
import pandas as pd



'''
----------------------------------------------------------------
Import data
'''
dataset_url = "https://raw.githubusercontent.com/harika-bonthu/02-linear-regression-fish/master/datasets_229906_491820_Fish.csv"

fish_df=pd.read_csv(dataset_url)
print(fish_df.head())  
print('------------------Import data-----------------------------')


'''
------------------------------------------------------------------
Check fish catagories and NAN data
'''
print(set(fish_df.Species)) # The 7 types of fishes
print(set(fish_df.Species.unique())) # The 7 types of fishes

print(fish_df.isna().sum()) # Chech whether there are missing values



'''
--------------------------------------------------------------------
Obtain data as X and Y
'''
features=fish_df.columns[1:]
X=fish_df[features].values
y=fish_df.iloc[:,0].values


'''
--------------------------------------------------------------------
Scale data by Maxminscalar
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X)
X_s = scaler.transform(X)



'''
--------------------------------------------------------------------
Encoder text as labels
'''
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(set(y))

print('------------------Data preprocessing-----------------------------')

'''
--------------------------------------------
Split training and testing data
'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_s, y, train_size=0.85, random_state=42)


'''
--------------------------------------------
Train the data and evaluate the performance

'''
from sklearn.linear_model import LogisticRegression


# Initialize 
log_reg=LogisticRegression(fit_intercept =True)   # Including the constant term

# Fit the model
log_reg.fit(X_train,y_train)


# Predict values
y_pre=log_reg.predict(X_test)
print(set(y_pre))
print(set(y_test))

print('------------------Model training and prediction-----------------------------')

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pre)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print('Feature coefficient:', log_reg.coef_)
print('intercept:',log_reg.intercept_)

'''
----------------------------------------------------------------------
Confusing matrix
'''
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pre)
plt.figure()
sns.heatmap(cf, annot=True)
plt.xlabel('Prediction')
plt.ylabel('Target')
plt.title('Confusion Matrix')
