""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question-6 (Lab-7)

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
from sklearn.cluster import DBSCAN 

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

# define the function of DBSCAN model for different epsilon value and minimum sample value
def dbscan_model(DATA,r_data, e, s):
    dbscan_model=DBSCAN(eps=e, min_samples=s).fit(r_data)
    DBSCAN_predictions = dbscan_model.labels_
    
    data_with_dbscan_clusters=r_data.copy()
    data_with_dbscan_clusters['Clusters']=DBSCAN_predictions
    
    # plot the data points with different colours for each cluster
    plt.scatter(data_with_dbscan_clusters['A'], data_with_dbscan_clusters['B'],c=data_with_dbscan_clusters['Clusters'], cmap='rainbow')
    plt.title(f'Scatter plot for epsilon= {e} and minimum samples= {s}')
    plt.xlabel('X - axis')
    plt.ylabel('Y - axis')
    plt.show()

    # find the purity score 
    print(f'Purity score for epsilon= {e} and minimum samples= {s} is = ',purity_score(y_true=DATA['Species'], y_pred=DBSCAN_predictions))
    print("------")

# call the DBSCAN model for different epsilon value and minimum sample value
dbscan_model(Data,reduced_data_dataframe,1,4)
dbscan_model(Data,reduced_data_dataframe,1,10)
dbscan_model(Data,reduced_data_dataframe,5,4)
dbscan_model(Data,reduced_data_dataframe,5,10)