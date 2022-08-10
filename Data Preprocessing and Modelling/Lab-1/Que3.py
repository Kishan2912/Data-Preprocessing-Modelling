""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for question 3 (Lab-1)

# import module from library
import pandas as pd

# reading the csv file using pandas
data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-1\lab assignment 1\pima-indians-diabetes.csv")

# (A) part
# Select the data of Age attribute
data1=data['Age'] 

# Select the data of all column excluding "class" and "Age" attribute
data2=data.iloc[:,[0,1,2,3,4,5,6]]

# list of column's refernce and their name
columns=list(data2) 
columnsname=list(data2.columns)
j=0

# Finding the correlation cofficient using pandas library
for i in columns:
    print(f"Correlation coefficient of Age attribute with {columnsname[j]} = ",data1.corr(data2[i]))
    j+=1
print("-------------------------")

# (B) part
# Select the data of BMI attribute
data1=data['BMI']

# Select the data of all column excluding "class" and "BMI" attribute
data2=data.iloc[:,[0,1,2,3,4,6,7]]

# list of column's refernce and their name
columns=list(data2)
columnsname=list(data2.columns) 
j=0

# Finding the correlation cofficient using pandas library
for i in columns:
    print(f"Correlation coefficient of BMI attribute with {columnsname[j]} = ",data1.corr(data2[i]))
    j+=1
