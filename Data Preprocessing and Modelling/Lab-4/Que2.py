""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 2 (Lab-4)

# import module from python library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

# read the csv file of spilting data of train and test set using pandas library
train_data=pd.read_csv('SteelPlateFaults-train.csv') 
test_data=pd.read_csv('SteelPlateFaults-test.csv') 

# Finding the value of x_train, x_test, y_train, y_test using appropiriate function
x_train=train_data.drop(columns=['Class'])
x_test=test_data.drop(columns=['Class'])
y_train=train_data['Class']
y_test=test_data['Class']

# Copy the x_train and x_test data
x_normalized_train = x_train.copy()
x_normalized_test= x_test.copy()

# normalization of train and test data using min-max normalization formula
for i in x_normalized_test.columns:
    x_normalized_test[i]=(x_normalized_test[i]-x_normalized_train[i].min())/(x_normalized_train[i].max()-x_normalized_train[i].min())  
for i in x_normalized_train.columns:
    x_normalized_train[i]=(x_normalized_train[i]-x_normalized_train[i].min())/(x_normalized_train[i].max()-x_normalized_train[i].min())   
   
# make the csv file of noramalized train and test data using to_csv function of csv library
x_normalized_train.to_csv('SteelPlateFaults-train-Normalised.csv') 
x_normalized_test.to_csv('SteelPlateFaults-test-normalised.csv')

# Part-a
# for knn classfier we use inbuilt function and using this classifier we fit the data and found the y_predection value
# find the confusion matrix for every k value using sklearn library inbuilt function confusion_matrix
print("Part-A ---> ")
K=[1,3,5]
for i in K:
    classifier=KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_normalized_train, y_train)
    y_pred=classifier.predict(x_normalized_test)
    cnf_matrix=confusion_matrix(y_test,y_pred)
    print(f"Confussion matrix for K={i} is --> \n ", cnf_matrix)

# Part-b
# find the accuracy of every k using sklearn library inbuilt function accuracy_score
print("Part-B ---> ")
for i in K:
    classifier=KNeighborsClassifier(n_neighbors=i)
    classifier.fit(x_normalized_train, y_train)
    y_pred=classifier.predict(x_normalized_test)
    accuracy=accuracy_score(y_test,y_pred)
    print(f"Classification accuracy for K={i} is :", accuracy)
print("-------------------------------------------------------------")