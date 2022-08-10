""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question-1 (Lab-7)

# import python libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA

# read the csv file using python library
Data=pd.read_csv("Iris.csv")
Data1=Data.iloc[:,[i for i in range(4)]]

# finding eigonvectors and eigonvalues of first four attribute using numpy library fuction
corr_matrix = Data1.corr()
eig_val, eig_vec = np.linalg.eig(corr_matrix)

# Plot eigon values of every attributes
plt.bar([0,1,2,3], eig_val,width=0.5,color="purple")
plt.xlabel('Position')
plt.title('Eigon value of attribute for PCA')
plt.ylabel("Eigon-value")
plt.show()

# finding 2 eigonvectors with highest eigonvalues using for loop
max_indices = []
for i in range(2):
    max = eig_val[0]
    maxi = 0
    for j in range(len(eig_val)):
        if eig_val[j] > max:
            max = eig_val[j]
            maxi = j
    max_indices.append(maxi)
    eig_val[maxi]=0

# find the eigon vector with highest and second highest eigonvalue
eig_vec_1 = eig_vec[:, max_indices[0]]  
eig_vec_2 = eig_vec[:, max_indices[1]] 

# reducing data into 2 dimension using PCA with Data1
pca_into_2 = PCA(n_components=2)
pca_into_2.fit(Data1)
reduced_data_into_2 = pca_into_2.fit_transform(Data1)
reduced_data_dataframe = pd.DataFrame(reduced_data_into_2, columns=['A', 'B'])

# ploting 2 dimensional data on scatter plot using matplotlib library
plt.scatter(reduced_data_dataframe['A'], reduced_data_dataframe['B'],color="red")
plt.title('Data projected along two dimensions')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.show()