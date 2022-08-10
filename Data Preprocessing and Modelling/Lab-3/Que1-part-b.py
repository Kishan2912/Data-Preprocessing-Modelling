""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 1 (Lab-3)

# import module from python library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-3\pima-indians-diabetes.csv")
data1=data.iloc[:,0:8]
columns_name=list(data1.columns)

# Removing the outliers
for i in columns_name:
    list1=list(data1[i])
    list1=sorted(list1)
    n=len(list1)
    Q1=data1[i].quantile(0.25)
    Q3=data1[i].quantile(0.75)
    IQR=Q3-Q1
    for j in range(len(data1[i])):
        if data1[i][j]<(Q1-1.5*IQR) or data1[i][j]>(Q3+1.5*IQR):
            data1.loc[j,i]=data1[i].median()

# Part-b
print("-Part-B")
old_mean=pd.DataFrame(data1.mean())
print("Mean value before standard normalization -->")
print(old_mean)
print()
old_stdev=pd.DataFrame(data1.std())
print("Standard dev value before standard normalization -->")
print(old_stdev)
print()

# Standard normalization
data_copy1=data1.copy()
for i in data_copy1.columns:
    for j in range(len(data_copy1[i])):
        mean=sum(data1[i])/len(data1[i])
        stdev=data1[i].std()
        data_copy1.loc[j,i]=(data_copy1[i][j]-mean)/stdev
        
print("Mean value after standard normalization -->")
new_mean=pd.DataFrame(data_copy1.mean())
print(new_mean)
print()
print("Standard dev value after standard normalization -->")
new_stdev=pd.DataFrame(data_copy1.std())
print(new_stdev)
print()