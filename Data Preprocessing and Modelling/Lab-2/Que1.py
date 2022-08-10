""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 1 (Lab-2)

# import module from python library
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file using pandas library
data = pd.read_csv('C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-2\landslide_data3_miss.csv')
columns=list(data.columns)

# count the missing value 
count_missingvalue=data.isnull().sum()
lis_missingvalue=list(data.isnull().sum())
print("---------Missing value in each attribute-------")
print(count_missingvalue)

# Plotting of missing value using matplotlib library
x_val=[i  for i in range(1,len(lis_missingvalue)+1)]
plt.bar(x_val, lis_missingvalue)
plt.xticks(x_val, columns)
plt.xlabel("Attribute Name")
plt.ylabel("Number of missing values in respective attribute")
plt.title("Plot of missing value in each attribute")
plt.show()
