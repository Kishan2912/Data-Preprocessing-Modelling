""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for question 6 (Lab-1)

# import module from library
import pandas as pd
import matplotlib.pyplot as plt

# reading the csv file using pandas
data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-1\lab assignment 1\pima-indians-diabetes.csv")

# Select the data of all column excluding "class" attribute
data1=data.iloc[:,[0,1,2,3,4,5,6,7]]

# list of column's refernce and their name
columns=list(data1)
columnsname=list(data1.columns)
j=0

# Plotting the box plot of every attribute using matplotlib.pyplot library
for i in columns:
    plt.figure()
    plt.boxplot(data1[i])
    plt.title(f"Box-plot of {columnsname[j]} attribute")
    j+=1
    plt.show()
