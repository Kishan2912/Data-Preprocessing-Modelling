""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for question 5 (Lab-1)

# import module from library
import pandas as pd
import matplotlib.pyplot as plt

# reading the csv file using pandas
data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-1\lab assignment 1\pima-indians-diabetes.csv")

# Select the data of pregs and class attribute
data1=data.iloc[:,[0,8]]

# Distribute the class attribute data using groupby function
groupdata=data1.groupby('class')

# Find the class 0 and class 1 data using get group function
class0=groupdata.get_group(0)
class1=groupdata.get_group(1)

# Plot the histogram of class 0 pregs attribute using matlpotlib.pyplot library
plt.title('Histogram plot for attribute pregs for class=0')
class0['pregs'].hist()
plt.show()

# Plot the histogram of class 1 pregs attribute using matlpotlib.pyplot library
plt.title('Histogram plot for attribute pregs for class=1')
class1['pregs'].hist()
plt.show()