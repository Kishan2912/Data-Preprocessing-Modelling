""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 5 (Lab-2)

# import module from python library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read the csv file using pandas library
miss_data = pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-2\landslide_data3_miss.csv")
new_data7=miss_data.loc[:, ['temperature', 'rain']].interpolate(method='linear', limit_direction='forward')
columns=list(new_data7.columns)

# Party-a
# Finding the outliers
outlier_list=[]
for i in columns:
    data_list=list(new_data7[i])
    data_list=sorted(data_list)
    n=len(data_list)
    if n%2==0:
        Q2=np.median(data_list)
        Q1=np.median(data_list[0:n//2+1])
        Q3=np.median(data_list[n//2-1:n])
    else:
        Q2=np.median(data_list)
        Q1=np.median(data_list[0:n//2+1])
        Q3=np.median(data_list[n//2:n])
    outlier_list.append((i,(Q1,Q2,Q3)))    
    IQR=Q3-Q1
    outlier_num=0
    for j in range(len(new_data7[i])):
        if new_data7[i][j]<(Q1-1.5*IQR) or new_data7[i][j]>(Q3+1.5*IQR):
            outlier_num+=1
    print("Number of outliers in", i, " before replacing outliers by median is : ",outlier_num) 

    # Plotting the box-plot
    plt.boxplot(new_data7[i])
    plt.title(f"Boxplot of attribute {i} before replacing outliers")
    plt.show()       

    # Part-b
    for j in range(len(new_data7[i])):
        if new_data7[i][j]<(Q1-1.5*IQR) or new_data7[i][j]>(Q3+1.5*IQR):
            outlier_num+=1
            new_data7[i][j]=new_data7[i].median()
    data_list=list(new_data7[i])
    data_list=sorted(data_list)
    n=len(data_list)
    if n%2==0:
        Q2=np.median(data_list)
        Q1=np.median(data_list[0:n//2+1])
        Q3=np.median(data_list[n//2-1:n])
    else:
        Q2=np.median(data_list)
        Q1=np.median(data_list[0:n//2+1])
        Q3=np.median(data_list[n//2:n])
    outlier_list.append((i,(Q1,Q2,Q3)))    
    IQR=Q3-Q1
    outlier_num=0
    for j in range(len(new_data7[i])):
        if new_data7[i][j]<(Q1-1.5*IQR) or new_data7[i][j]>(Q3+1.5*IQR):
            outlier_num+=1
    print("Number of outliers in", i, " after replacing outliers by median is : ",outlier_num) 

    # Plotting the box-plot
    plt.boxplot(new_data7[i])
    plt.title(f"Boxplot of attribute {i} After replacing outliers")
    plt.show()   
    print("--------------")    