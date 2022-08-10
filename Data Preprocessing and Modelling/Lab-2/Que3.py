""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 3 (Lab-2)

# import module from python library
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file using pandas library
data = pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-2\landslide_data3_miss.csv")

new_data1=data.dropna(subset=['stationid'])
new_data2=new_data1.dropna(thresh=7)

print("-------------Missing value in each attribute----------")
count_nullvalue=new_data2.isnull().sum()
print(count_nullvalue)

print("Total missing value in data after tuple deleting is : ",new_data2.isnull().sum().sum())