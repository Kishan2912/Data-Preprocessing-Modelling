""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question-2 (Lab-7)

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

# apply the k-means on reducing dataframe
kmeans = KMeans(n_clusters=3) 
kmeans.fit(reduced_data_dataframe)
kmeans_prediction = kmeans.predict(reduced_data_dataframe)
kmeans_cen_x=[]
kmeans_cen_y=[]
for i in kmeans.cluster_centers_:
    kmeans_cen_x.append(i[0])
    kmeans_cen_y.append(i[1])

# put the prediction of kmeans culster in copy dataframe
data_kmeans_clusters=reduced_data_dataframe.copy()
data_kmeans_clusters['Clusters']=kmeans_prediction

# part a
# Plot the data points with different colours for each cluster of kmeans culster dataframe
plt.scatter(data_kmeans_clusters['A'], data_kmeans_clusters['B'],c=data_kmeans_clusters['Clusters'], cmap='rainbow')
plt.scatter(kmeans_cen_x,kmeans_cen_y, marker='X', color='black', label='Cluster center')
plt.title('Kmeans different cluster for K=3')
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.legend()
plt.show()

# part b
# find the sum of squared distances of samples to their closest cluster centre
kmeans_features_value=reduced_data_dataframe.values
kmeans_distortion=kmeans.inertia_
print('Distortion measure is = ',kmeans_distortion)
print("---------------")

# part c
# find the Purity score
print('Purity score is = ',purity_score(y_true=Data['Species'], y_pred=kmeans_prediction))
print("--------------------------------------------------------------------")