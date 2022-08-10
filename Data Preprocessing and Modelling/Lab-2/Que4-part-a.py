""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 4 (Lab-2)

# import module from python library
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file using pandas library
miss_data = pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-2\landslide_data3_miss.csv")
orignal_data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-2\landslide_data3_original.csv")
miss_data1=miss_data.iloc[:,2:9]

new_data1=miss_data.dropna(subset=['stationid'])
new_data2=new_data1.dropna(thresh=7)
new_data3=new_data2.fillna(0)
new_data4=new_data3.iloc[:,2:9]

# fill the mean value at the missing places using fillna function
val=new_data4.mean()
val=pd.DataFrame(val)
values = {"temperature" : val[0][0], "humidity" : val[0][1], "pressure" : val[0][2], "rain" : val[0][3], "lightavgw/o0" : val[0][4],"lightmax" : val[0][5],"moisture" : val[0][6]}
new_data6=miss_data1.fillna(value=values)

# Part-a (i)
# Find the static value for mean filled file
filled_col=list(new_data6.columns)
filled_mean=list(new_data6.mean())
filled_median=list(new_data6.median())
filled_mode=list(new_data6.mode().loc[0])
filled_std=list(new_data6.std())

new_data7=[(filled_col[i],filled_mean[i], filled_median[i], filled_mode[i], filled_std[i]) for i in range(len(filled_col))]
filled_static=pd.DataFrame(new_data7,  columns=[ 'Atributes','Mean','Median','Mode', 'Stdev'])
print("-----Part-A(i)-------")
print("----Value of mean, median, mode and stdev after filling mean in Each attribute ---")
print(filled_static)
print()

# Find the static value for original file
orignal_data6=orignal_data.iloc[:,2:9]
orignal_col=list(orignal_data6.columns)
orignal_mean=list(orignal_data6.mean())
orignal_median=list(orignal_data6.median())
orignal_mode=list(orignal_data6.mode().loc[0])
orignal_std=list(orignal_data6.std())


orignal_data7=[(orignal_col[i],orignal_mean[i], orignal_median[i], orignal_mode[i], orignal_std[i]) for i in range(len(orignal_col))]
original_static=pd.DataFrame(orignal_data7,  columns=[ 'Atributes','Mean','Median','Mode', 'Stdev'])
print("----Value of mean, median, mode and stdev of orignal file ---")
print(original_static)
print()

# Part-a (ii)
# Find the RMSE value of different attribute
rmse_list=[]
rmse_value=[]
attri_name=[]
for i in filled_col:
    rmse_2=0
    new_list=list(new_data6.loc[:,i])
    original_list=list(orignal_data6.loc[:,i])
    for j in range(len(new_list)):
        rmse_2+=(new_list[j]-original_list[j])**2
    rmse=(rmse_2/miss_data[i].isnull().sum())**0.5
    rmse_list.append((i,rmse))
    rmse_value.append(rmse)
    attri_name.append(i)
rmse_dataframe=pd.DataFrame(rmse_list, columns=['Atribute Name', 'RMSE value'])
print("-----Part-A(ii)-------")
print("------RMSE value of different Attribute--------")
print(rmse_dataframe)

# Plot the RMSE value of attribute
x_axis=[i for i in range(1,len(rmse_list)+1)]
plt.bar(x_axis, rmse_value)
plt.xticks(x_axis, attri_name)
plt.xlabel("Attribute Name")
plt.ylabel("RMSE value")
plt.show()

