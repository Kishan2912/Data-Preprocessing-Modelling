""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question-3 (Lab-7)

# import python libraries
from numpy.lib import r_
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
import scipy 
from sklearn import metrics 
from scipy.optimize import linear_sum_assignment

# read the csv file using python library
Data=pd.read_csv("Iris.csv")
Data1=Data.iloc[:,[i for i in range(4)]]

# reducing data into 2 dimension using PCA with Data1
pca_into_2 = PCA(n_components=2)
pca_into_2.fit(Data1)
reduced_data_into_2 = pca_into_2.fit_transform(Data1)
reduced_data_dataframe = pd.DataFrame(reduced_data_into_2, columns=['A', 'B'])

# define the purity score function
def purity_score(y_true, y_pred): 
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred) 
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

# different value of K
K=range(2,8)
# define the distortion measure and purity score list
kmeans_dis=[]
kmeans_purity_score=[]
# find the distortion measure and purity score for different K-value using kmeans model
for k in K:
    kmeans = KMeans(n_clusters=k) 
    kmeans.fit(reduced_data_dataframe)
    kmeans_prediction = kmeans.predict(reduced_data_dataframe)
    kmeans_features_value=reduced_data_dataframe.values
    kmeans_distortion=kmeans.inertia_
    kmeans_dis.append(kmeans_distortion)
    kmeans_purity_score.append(purity_score(y_true=Data['Species'], y_pred=kmeans_prediction))

# print the distortion measure for different K-value
print("List of distortion measure :",kmeans_dis)
print("-----------")

# plot Number of clusters(K) vs. distortion measure 
plt.plot(K, kmeans_dis, marker='o',color='red')
plt.xlabel('Number of cluster (K)')
plt.ylabel('Distortion measure for different K-value')
plt.title('Number of clusters(K) vs. distortion measure')  
plt.show()

# print the purity score for different K-value
print('List of purity score :',kmeans_purity_score)  
print("--------------------------------------------")