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

# Part-a
print("-Part-A-")
old_max=pd.DataFrame(data1.max())
print("Maximum value before min-max normalization -->")
print(old_max)
print()
print("Minimum value before min-max normalization -->")
old_min=pd.DataFrame(data1.min())
print(old_min)
print()

# Min-max normaliztion of data
data_copy1=data1.copy()
for i in data_copy1.columns:
    for j in range(len(data_copy1[i])):
        data_copy1.loc[j,i]=((data_copy1[i][j]-min(data_copy1[i]))/(max(data_copy1[i])-min(data_copy1[i])))*7+5

print("Maximum value after min-max normalization -->")
new_max=pd.DataFrame(data_copy1.max())
print(new_max)
print()
print("Minimum value after min-max normalization -->")
new_min=pd.DataFrame(data_copy1.min())
print(new_min)
print()

