# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 20:55:28 2022

@author: Yuanyuan_Tang
Linear regression with regularization
"""



'''
Import libraries
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston



'''
------------------------------------------------------------------------------------
Import data and simple preprocessing
'''
boston=load_boston()
boston_df=pd.DataFrame(boston.data, columns=boston.feature_names)  # Import the data
boston_df['Price']=boston.target
print(np.shape(boston_df.columns))   # The number of columns


print(boston_df.head()) # Show the data


'''
 SHow the covariance
'''
plt.figure(figsize = (10, 10))
sns.heatmap(boston_df.corr(), annot = True)



#There are cases of multicolinearity, we will drop a few columns that are not linear
boston_df.drop(columns = ["INDUS", "NOX"], inplace = True)  # Drop columns of DataFrame

#pairplot: show the distribution of two pair of variables, which can be used to present
#   the relationship between features and targets
#sns.pairplot(boston_df)

#we will log the LSTAT Column
#boston_df.LSTAT = np.log(boston_df.LSTAT)
boston_df["LSTAT"] = np.log(boston_df["LSTAT"])


'''
------------------------------------------------------------------------------------
Data splitting and scaling
Data splitting: split data into training and testing data sets
'''

#preview
print('The number of features are:',np.shape(boston_df.columns))
features = boston_df.columns[0:11]   # Acquire the names of features
print('The number of features are:',np.shape(features))
target = boston_df.columns[-1]     # Acquire the name of the target

#X and y values
X = boston_df[features].values
y = boston_df[target].values

#splot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12)

print("The dimension of X_train is {}".format(X_train.shape))
print("The dimension of X_test is {}".format(X_test.shape))




'''
--------------------------------------------------------------------------
Scale features following standard distribution
'''

#Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print('--------------Data splitting and rescale------------------------')

'''
--------------------------------------------------------------------------
Training Linear Regression Models and evaluate models by R2
'''

# Import linear regression model
lr=LinearRegression()

# Train models
lr.fit(X_train, y_train)

# Predict
y_t_pre=lr.predict(X_test)



#actual
actual = y_test

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

print("The train score for lr model is {}", train_score_lr)
print("The test score for lr model is {}", test_score_lr)


# MSE: verify whether they are quivalent
from sklearn.metrics import r2_score
test_mse_lr=r2_score(y_test,y_t_pre)

print("The test R2 for lr model is {}", test_mse_lr)
print('---------------linear model----------------------------')


'''
--------------------------------------------------------------------------
Training Lasso Models and evaluate models by R2

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
'''
# Import lasso
lr_lasso=Lasso(alpha=1)


# Train model
lr_lasso.fit(X_train, y_train)

# Predict
y_t_lrlasso=lr_lasso.predict(X_test)


# Model evaluation
test_score_lrlasso = lr_lasso.score(X_test, y_test)
print("The test score for lr_lasso model is {}", test_score_lrlasso )


# MSE: verify whether they are quivalent
#from sklearn.metrics import r2_score
test_mse_lrlasso=r2_score(y_test,y_t_lrlasso)

print("The test R2 for lr_lasso model is {}", test_mse_lrlasso)

print('-----------------lasso model--------------------------')


'''
--------------------------------------------------------------------------
Training Ridge Models and evaluate models by R2

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
'''
# Import lasso
lr_ridge=Ridge(alpha=10)


# Train model
lr_ridge.fit(X_train, y_train)

# Predict
y_t_lrridge=lr_ridge.predict(X_test)


# Model evaluation
test_score_lrridge = lr_ridge.score(X_test, y_test)
print("The test score for lr_ridge model is {}", test_score_lrridge )


# MSE: verify whether they are quivalent
#from sklearn.metrics import r2_score
test_mse_lrridge=r2_score(y_test,y_t_lrridge)

print("The test R2 for lr_ridge model is {}", test_mse_lrridge)

print('-----------------ridge model--------------------------')


'''
--------------------------------------------------------------------------
Training ElasticNet Models and evaluate models by R2

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
'''
# Import lasso
lr_ElasticNet=ElasticNet(alpha=0.5, l1_ratio=0.1) #Alpha increase both, l1=0, Ridge; l1=1, Lasso


# Train model
lr_ElasticNet.fit(X_train, y_train)

# Predict
y_t_lrElasticNet=lr_ElasticNet.predict(X_test)


# Model evaluation
test_score_lrElasticNet = lr_ElasticNet.score(X_test, y_test)
print("The test score for lr_ElasticNet model is {}", test_score_lrElasticNet )


# MSE: verify whether they are quivalent
#from sklearn.metrics import r2_score
test_mse_lrElasticNet=r2_score(y_test,y_t_lrElasticNet)

print("The test R2 for lr_ElasticNet model is {}", test_mse_lrElasticNet)

print('-----------------ElasticNet model--------------------------')


'''
--------------------------------------------------------------------------
Comparing 4 models by R2
'''
Model_names=['lr','lr_ridge','lr_lasso','lr_elasticNet']
R2_models=[test_mse_lr,test_score_lrlasso,test_mse_lrridge, test_mse_lrElasticNet ]

plt.figure(figsize=(10,10))
plt.bar(Model_names,R2_models, color=['blue','red','green','pink'])
plt.legend('R_squre')



plt.figure(figsize=(10,10))
plt.plot(features,lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
plt.plot(features,lr_lasso.coef_,alpha=0.4,linestyle='none',marker='*',markersize=7,color='red',label='Lasso')
plt.plot(features,lr_ridge.coef_,alpha=0.4,linestyle='none',marker='d',markersize=7,color='blue',label='Ridge')
plt.plot(features,lr_ElasticNet.coef_,alpha=0.4,linestyle='none',marker='^',markersize=7,color='black',label='ElasticNet')


#rotate axis
plt.xticks(rotation = 90)
plt.legend()
plt.title("Comparison of 4 linear regression models")
plt.show()