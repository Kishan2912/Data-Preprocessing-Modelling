""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Part-B Que-1 (Lab-5)

# import module from python library
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# # read the csv file of marine snails data using pandas library
data=pd.read_csv('abalone.csv')

# Spilting the data into train (70%) and test (30%) using train_test_split function and random size value is 42
train_data, test_data = train_test_split(data, train_size=0.7, random_state=42, shuffle=True)

# store the train and test data in csv file using to_csv funtion of csv library
train_data.to_csv('abalone-train.csv')
test_data.to_csv('abalone-test.csv') 

# find the correlation matrix of data using corr function
corr_matrix=data.corr()  

# from the correlation matrix, we found that "Shell weight" has maximum correlation with target attribute Rings

# define the linear regression model for testing data using train data as input
def linear_rgg(x_train,y_train,x_test,y_test,t_t):
    x_train=pd.DataFrame(x_train)
    y_train=pd.DataFrame(y_train)
    x_test=pd.DataFrame(x_test)
    y_test=pd.DataFrame(y_test)
    lin_reg=LinearRegression()
    model=lin_reg.fit(x_train,y_train)
    if t_t==0:
        pred=model.predict(x_train)
    elif t_t==1:
        pred=model.predict(x_test)
    pred_list=[]
    for i in range(len(pred)):
        pred_list.append(pred[i][0])
    return(pred_list)

# define the function for root mean square error value
def root_mean_squared_error(x,y):
    rmse=0
    x=np.array(x)
    y=np.array(y)
    for i in range(len(x)):
        rmse+=(x[i]-y[i])**2
    return (rmse/len(x))**0.5

# Que 1 part a
# scatter plot of training data
plt.scatter(train_data['Shell weight'],train_data['Rings'], marker='x', label="Actual rings", color='red')
plt.title("Best fit line on training data")
plt.xlabel('Shell weight')
plt.ylabel('Rings')
# draw the fit line on scatter plot of taining data
plt.plot(train_data['Shell weight'], linear_rgg(train_data['Shell weight'],train_data['Rings'],test_data['Shell weight'],test_data['Rings'],0), color ='yellow', label='Predicted rings')
plt.legend()
plt.show()

# Que 1 part b
print("Que 1 Part-b :")
print("Prediction accuracy on the training data is :", root_mean_squared_error(train_data['Rings'],linear_rgg(train_data['Shell weight'],train_data['Rings'],train_data['Shell weight'],train_data['Rings'],0)))
print("------")

# Que 1 part c
print("Que 1 Part-c :")
print("Prediction accuracy on the testing data is :", root_mean_squared_error(test_data['Rings'],linear_rgg(train_data['Shell weight'],train_data['Rings'],test_data['Shell weight'],test_data['Rings'],1)))
print("------")

# Que 1 part d
plt.scatter(test_data['Rings'],linear_rgg(train_data['Shell weight'],train_data['Rings'],test_data['Shell weight'],test_data['Rings'],1), marker='o')
plt.title("For testing data actual v/s predection plot")
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.show()
