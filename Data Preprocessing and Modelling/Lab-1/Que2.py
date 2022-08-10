""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for question 2 (Lab-1)

# import module from library
import pandas as pd
import matplotlib.pyplot as plt

# reading the csv file using pandas
data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-1\lab assignment 1\pima-indians-diabetes.csv")

# (A) Part
# Select the data of Age attribute
data1=data['Age'] 

# Select the data of all column excluding "class" and "Age" attribute
data2=data.iloc[:,[0,1,2,3,4,5,6]]

# list of column's refernce and their name
columns=list(data2) 
columnsname=list(data2.columns)
j=0

# Plotting of every scatter plot using matplotlib.pyplot library
for i in columns:
    plt.scatter(data1,data2[i],c='red')
    plt.title(f"Scatter plot of Age attribute with {columnsname[j]}", fontsize=20)
    plt.xlabel("Value of Age attribute")
    plt.ylabel(f"Value of {columnsname[j]} attribute")
    plt.show()
    j+=1
    
# (B) part
# Select the data of BMI attribute
data1=data['BMI']

# Select the data of all column excluding "class" and "BMI" attribute
data2=data.iloc[:,[0,1,2,3,4,6,7]]

# list of column's refernce and their name
columns=list(data2)
columnsname=list(data2.columns)
j=0

# Plotting of every scatter plot using matplotlib.pyplot library
for i in columns:
    plt.scatter(data1,data2[i],c='blue')
    plt.title(f"Scatter plot of BMI attribute with {columnsname[j]}",fontsize=20)
    plt.xlabel("Value of BMI attribute")
    plt.ylabel(f"Value of {columnsname[j]} attribute")
    plt.show()
    j+=1