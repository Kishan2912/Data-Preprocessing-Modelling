""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Part-B Que-4 (Lab-5)

# import module from python library
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# read the csv file of marine snails data using pandas library
data=pd.read_csv('abalone.csv')

# Spilting the data into train (70%) and test (30%) using train_test_split function and random size value is 42
train_data, test_data = train_test_split(data, train_size=0.7, random_state=42, shuffle=True)

# store the train and test data in csv file using to_csv funtion of csv library
train_data.to_csv('abalone-train.csv')
test_data.to_csv('abalone-test.csv') 

# find the correlation matrix of data using corr function
corr_matrix=data.corr()  

# from the correlation matrix, we found that "Shell weight" has maximum correlation with target attribute Rings

# define the multi non linear regression model for testing data using train data as a input
def mul_non_linear_rgg(x_train,y_train,x_test,y_test,p,t_t):
    poly_features = PolynomialFeatures(degree=p)
    x_poly = poly_features.fit_transform(x_train)
   
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    if t_t==0:
        pred = regressor.predict(poly_features.fit_transform(x_train))
    elif t_t==1:
        pred = regressor.predict(poly_features.fit_transform(x_test))
    return pred

# define the function for root mean square error value
def root_mean_squared_error(x,y):
    rmse=0
    x=np.array(x)
    y=np.array(y)
    for i in range(len(x)):
        rmse+=(x[i]-y[i])**2
    return (rmse/len(x))**0.5

# store the p-value in a list
P=[2,3,4,5]

# Que 4 part a
print("Que 4 Part-a :")
# find the RMSE value of training data using for loop
rmse_value_training=[]
for i in P:
    pred=mul_non_linear_rgg(train_data.iloc[:,0:7],train_data['Rings'],test_data.iloc[:,0:7],test_data['Rings'],i,0)
    rmse_value_training.append(root_mean_squared_error(np.array(train_data['Rings']), pred))
print("Prediction accuracy on the training data for P=2,3,4,5 are :", rmse_value_training)
print() 
# plot the Rmse value of training data
plt.bar(P, rmse_value_training)
plt.xlabel("Degree of the polynomial")
plt.ylabel("RMSE")
plt.title("For training data RMSE value")
plt.show()
print("-----")

# Que 4 part B
print("Que 4 Part-B :")
# find the RMSE value of training data using for loop
rmse_value_testing=[]
for i in P:
    pred=mul_non_linear_rgg(train_data.iloc[:,0:7],train_data['Rings'],test_data.iloc[:,0:7],test_data['Rings'],i,1)
    rmse_value_testing.append(root_mean_squared_error(np.array(test_data['Rings']), pred))
print("Prediction accuracy on the testing data for P=2,3,4,5 are :", rmse_value_testing)
# plot the Rmse value of testing data
plt.bar(P, rmse_value_testing)
plt.xlabel("Degree of the polynomial")
plt.ylabel("RMSE")
plt.title("For testing data RMSE value")
plt.show()
print("-----")

# Que 4 part C
# predect the test value
test_pred=mul_non_linear_rgg(train_data.iloc[:,0:7],train_data['Rings'],test_data.iloc[:,0:7],test_data['Rings'],2,1)
# draw the scatter plot of testing data actual rings v/s predicted rings
plt.scatter(test_data['Rings'], test_pred ,marker='o')
plt.title("For testing data actual rings v/s predicted rings")
plt.xlabel('Actual rings')
plt.ylabel("Predicted rings")
plt.show()