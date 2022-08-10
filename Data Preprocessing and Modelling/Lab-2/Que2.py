""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 2 (Lab-2)

# import module from python library
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file using pandas library
data = pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-2\landslide_data3_miss.csv")

# Part-a
new_data1=data.dropna(subset=['stationid'])
print("----------Part-a-------")
print("Rows in old data(before delitation) : ",len(data))
print("Rows in new data(after delitation) : ",len(new_data1))
print("Number of tuple deleted : ", len(data)-len(new_data1))

# Part-b
new_data2=new_data1.dropna(thresh=7)
print("----------Part-b-------")
print("Rows in old data(before delitation) : ",len(new_data1))
print("Rows in new data(after delitation) : ",len(new_data2))
print("Number of tuple deleted : ", len(new_data1)-len(new_data2))

  