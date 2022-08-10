""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 1 (Lab-1)

# import module from library
import pandas as pd

# reading the csv file using pandas
data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-1\lab assignment 1\pima-indians-diabetes.csv")

# Select the data of all column excluding "class" attribute
data1=data.iloc[:,[0,1,2,3,4,5,6,7]] 

# finding the all statical parameters using pandas library
print("----Mean of all the attributes excluding 'class' attribute ----- ")
print(data1.mean())
print()

print("----Median of all the attributes excluding 'class' attribute ----- ")
print(data1.median())
print()

print("----Mode of all the attributes excluding 'class' attribute ----- ")
print(data1.mode())
print()

print("----Maximum of all the attributes excluding 'class' attribute ----- ")
print(data1.max())
print()

print("----Minimum of all the attributes excluding 'class' attribute ----- ")
print(data1.min())
print()

print("----Standard deviation of all the attributes excluding 'class' attribute ----- ")
print(data1.std())
print()