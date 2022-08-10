""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question-4 (Lab-7)

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

# use the GMM model to reduced the data into 3-dimensional data
gmm = GaussianMixture(n_components = 3)
gmm.fit(reduced_data_dataframe)
GMM_prediction = gmm.predict(reduced_data_dataframe)
data_with_gmm_clusters=reduced_data_dataframe.copy()
data_with_gmm_clusters['Clusters']=GMM_prediction

# part a
gmm_features_value=reduced_data_dataframe.values
gmm_centers = np.empty(shape=(gmm.n_components, gmm_features_value.shape[1]))
for i in range(gmm.n_components):
    density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(reduced_data_dataframe)
    gmm_centers[i, :] = gmm_features_value[np.argmax(density)]

# Plot the data points with different colours for each cluster
plt.scatter(data_with_gmm_clusters['A'], data_with_gmm_clusters['B'],c=data_with_gmm_clusters['Clusters'], cmap='rainbow')
plt.scatter(gmm_centers[:, 0], gmm_centers[:, 1], s=20, marker='x', color='black', label='Cluster center')
plt.title('nGMM culster for K=3')
plt.legend()
plt.xlabel('X - axis')
plt.ylabel('Y - axis')
plt.show()

# part b
# find the distortion measure
gmm_distortion=sum(gmm.score_samples(reduced_data_dataframe))
print('Distortion measure is = ',gmm_distortion)
print("---------")

# part c
# find the Purity score
print('Purity score is = ',purity_score(y_true=Data['Species'], y_pred=GMM_prediction))
print("---------------------------------------")