""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 1 (Lab-4)

# import module from python library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

# read the csv file of given data using pandas library
data=pd.read_csv("SteelPlateFaults-2class.csv")

# spilt the data according to class using log function and store the data
data_1=data.loc[data['Class']==1]
data_0=data.loc[data['Class']==0]

# spilting the data into train and test set using train_test_split function with each of class and random state = 42
# Combine the both class data using concat function for train and test set
[train_1,test_1]=train_test_split(data_1, train_size=0.7, random_state=42,shuffle=True)
[train_0,test_0]=train_test_split(data_0, train_size=0.7, random_state=42,shuffle=True)
train_data=pd.DataFrame(pd.concat([train_1,train_0]))
test_data=pd.DataFrame(pd.concat([test_1,test_0]))

# make the csv file of train and test data using to_csv function of csv library
train_data.to_csv('SteelPlateFaults-train.csv', index=False) 
test_data.to_csv('SteelPlateFaults-test.csv', index=False) 
 
# Finding the value of x_train, x_test, y_train, y_test using appropiriate function
x_train=train_data.drop(columns=['Class'])
x_test=test_data.drop(columns=['Class'])
y_train=train_data['Class']
y_test=test_data['Class']


# Part-a
# for knn classfier we use inbuilt function and using this classifier we fit the data and found the y_predection value
# find the confusion matrix for every k value using sklearn library inbuilt function confusion_matrix
print("Part-A ---> ")
K=[1,3,5]
for i in K:
    classifier=KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    y_pred=classifier.predict(x_test)
    cnf_matrix=confusion_matrix(y_test,y_pred)
    print(f"Confussion matrix for K={i} is --> \n", cnf_matrix)
print()

# Part-b
# find the accuracy of every k using sklearn library inbuilt function accuracy_score
print("Part-B ---> ")
for i in K:
    classifier=KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_train, y_train)
    y_pred=classifier.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Classification accuracy for K={i} is :", accuracy)
print("--------------------------------------------------------------------------")