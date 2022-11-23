# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:42:39 2022

@author: Yuanyuan_Tang

Decision tree



"""

'''
-------------------------------------------------------
Import module
'''

from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.model_selection import train_test_split


'''
------------------------------------
Generate a n-class problem
'''

X, t = make_classification(100, 5, n_classes=2, shuffle=True, random_state=10)


'''
---------------------------------------
train models and predict values 
Shuffle=True: prevent correlation among data
'''
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, shuffle=True, random_state=1) 

model = tree.DecisionTreeClassifier()
model = model.fit(X_train, t_train)


tree.plot_tree(model)


predicted_value = model.predict(X_test)
print(predicted_value)


'''
---------------------------------------
Gini parameter: a score in the range [0,1] that evaluates how accurate a split is among the classified groups. 
'''


zeroes = 0
ones = 0
for i in range(0, len(t_train)):
	if t_train[i] == 0:
		zeroes += 1
	else:
		ones += 1

print(zeroes)
print(ones)

val = 1 - ((zeroes/70)*(zeroes/70) + (ones/70)*(ones/70))
print("Gini :", val)



'''
------------------------------------
Predict accuracy
'''

match = 0
UnMatch = 0

for i in range(30):
	if predicted_value[i] == t_test[i]:
		match += 1
	else:
		UnMatch += 1

accuracy = match/30
print("Accuracy is: ", accuracy)
