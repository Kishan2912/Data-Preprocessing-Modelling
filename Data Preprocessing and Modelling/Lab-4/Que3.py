""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 3 (Lab-4)

# import module from python library
import pandas as pd 
import numpy as np
from pandas.core.tools.datetimes import Scalar
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# read the csv file of spilting data of train and test set using pandas library
train_data=pd.read_csv('SteelPlateFaults-train.csv') 
test_data=pd.read_csv('SteelPlateFaults-test.csv') 

# remove the some data column using drop function
train_data.drop(columns=['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],inplace=True)
test_data.drop(columns=['TypeOfSteel_A300','TypeOfSteel_A400','X_Minimum','Y_Minimum'],inplace=True)

# Classifing the train data basis on classes
train_data_0=train_data.loc[train_data['Class']==0]
train_data_1=train_data.loc[train_data['Class']==1]

# Classifing the test data basis on classes
test_data_0=train_data.loc[train_data['Class']==0]
test_data_1=train_data.loc[train_data['Class']==1]

# Finding the value of x_train_0, x_train_1, x_test, y_train, y_test using appropiriate function
x_train_0=train_data_0.drop(columns=['Class'])
x_train_1=train_data_1.drop(columns=['Class'])
x_test=test_data.drop(columns=['Class'])
y_train=train_data['Class']
y_test=test_data['Class']

# find the mean vector of both classes using mean function of numpy library
mean_vec_0=np.mean(x_train_0)
mean_vec_1=np.mean(x_train_1)

# find the covarince matrix of both classes using cov function of numpy library
cov_0=np.cov(x_train_0.T)
cov_1=np.cov(x_train_1.T)
print("Covariance matrix of class-0 : \n", cov_0)
print("Covariance matrix of class-1 : \n", cov_1)

# define the likelihood function using the expression of multivariate Gaussian density
def likelihood(x, mean, cov):
    power = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    return (np.exp(power)) / ((2 * np.pi) * 5 * (np.linalg.det(cov)) * 0.5)

# Finding the priority of class 1 and class 2 using basic formula
prior_0=len(train_data_0)/(len(train_data_0)+len(train_data_1))
prior_1=len(train_data_1)/(len(train_data_0)+len(train_data_1))

# apply the train model on test data and pridect the output 
# find the probability of test data for both classes
y_pred=[]
for index,row in x_test.iterrows():
    p_0=likelihood(row,mean_vec_0,cov_0)*prior_0
    p_1=likelihood(row,mean_vec_1,cov_1)*prior_1
    if(p_0>p_1):
        y_pred.append(0)
    else:
        y_pred.append(1)

# find the accuracy of bayes classifier using accuracy score function of sklearn library
bay_acc=accuracy_score(y_test,y_pred)

print("Confusion matrix of data -->  \n",confusion_matrix(y_test,y_pred))
print("Accuracy of Bayes classifier: ",bay_acc)        