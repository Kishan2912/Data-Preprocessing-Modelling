""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 3 (Lab-3)

# import module from python library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

old_mean=pd.DataFrame(data1.mean())
old_stdev=pd.DataFrame(data1.std())
# Standard normalization
data_copy1=data1.copy()
for i in data_copy1.columns:
    for j in range(len(data_copy1[i])):
        mean=sum(data1[i])/len(data1[i])
        stdev=data1[i].std()
        data_copy1.loc[j,i]=(data_copy1[i][j]-mean)/stdev

# Part-a
# Reduce the multidimensional (d = 8) data into lower dimensions (l = 2)
print("Part-A -")
pca=PCA(n_components=2)
pca.fit(data_copy1)
projected_data=pca.fit_transform(data_copy1)
projected_data_dataframe = pd.DataFrame(projected_data, columns=['A', 'B'])

# Variance of the projected data
print("Variance of the projected data")
print(projected_data_dataframe.var())
print()

# finding eigenvectors
corr_matrix=data_copy1.corr()
eig_val,eig_vec=np.linalg.eig(corr_matrix)
print("Two maximum eigen value -->")
max_index = []
for i in range(2):
    max = eig_val[0]
    maxi = 0
    for j in range(len(eig_val)):
        if eig_val[j] > max:
            max = eig_val[j]
            maxi = j
    print("Max eigned values is ", max,'and it is found at',maxi+1)        
    max_index.append(maxi)
    eig_val[maxi]=0
print()

# Scatter plot of projected data
plt.scatter(projected_data_dataframe['A'],projected_data_dataframe['B'])
plt.title("Projected data along 2-dimensios")
plt.xlabel("X-value")
plt.ylabel("Y-value")
plt.show()


# Part-b
# Sorting and plotting of eigon values
eig_val,eig_vec=np.linalg.eig(corr_matrix)
eig_val = list(eig_val)
eig_val.sort(reverse=True)
print(eig_val)
x_val = [i+1 for i in range(len(eig_val))]
plt.bar(x_val, eig_val)
plt.title('Eigen Values plotting in Descending Order')
plt.xlabel('Position of eigen value')
plt.ylabel('Eigen Value')
plt.show()

# Part-c
# Finding the reconstructed error
print("Part-C -")
rec_error= []
for i in range(1, 9):
    pca = PCA(n_components=i)
    projected_data = pca.fit_transform(data_copy1)
    reconstructed_data = pca.inverse_transform(projected_data)
    rec_error.append(np.linalg.norm((data_copy1-reconstructed_data), None))

# Plot of reconstruction error
x = [i for i in range(1,len(rec_error)+1)]
plt.plot(x, rec_error)
plt.title('Plot of reconstruction error in Euclidean distance for l=1 to 8')
plt.xlabel('Value of l dimensions')
plt.ylabel('Euclidian distance')
plt.show()

# Finding the covariance matrix
for i in range(1, 9):
    pca = PCA(n_components=i)
    projected_data = pca.fit_transform(data_copy1)
    new_data=pd.DataFrame(projected_data)
    print(f'Covariance matrix for {i} dimension -->')
    print(new_data.cov())
print()

# Part-d
print("Part-D -")
print("Covariance matrix of original data -->")
print(corr_matrix)   
pca_8 = PCA(n_components=8)
pca_8_result = pca_8.fit_transform(data_copy1)
new_8=pd.DataFrame(pca_8_result)
print("Covariance matrix of new data -->")
print(new_8.corr())
print() 