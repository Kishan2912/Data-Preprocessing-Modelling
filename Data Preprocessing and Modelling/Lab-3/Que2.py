""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question 2 (Lab-3)

# import module from python library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Giving value
mean=[0,0]
cov_matrix=[[13,-3],[-3,5]]

# Generation of data
D=np.random.multivariate_normal(mean,cov_matrix,1000)
print("Generate 2-dimensional synthetic data -->")
print(D)
print()

# Part-a
# Scatter plot of data
plt.scatter(D[:,[0]],D[:,[1]],label='Genrerated data points')
plt.title("Scatter plot of genreted samples")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# Part-b
#Calculate Eigen values and Eign Vectors
eig_val,eig_vec=np.linalg.eig(cov_matrix)
print("Eigonvalues of covariance matrix --> ")
print(eig_val)
print()
print("Eigonvectors of covariance matrix --> ")
print(eig_vec)
print()

# Plotting the Eigen directions onto the scatter plot of data
origin = [0, 0]
eig_vec1 = eig_vec[:,0]
eig_vec2 = eig_vec[:,1]
plt.scatter(D[:,[0]],D[:,[1]],label='Genrerated data points')
plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
plt.quiver(*origin, *eig_vec2, color=['b'], scale=10)
plt.title("Eigonvectors direction on scatter plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

# Part-c
# Projection data plotting
E=np.dot(D,eig_vec)
for i in range(2):
    x_value=[]
    y_value=[]
    for d in E:
        x_value.append(d[i]*eig_vec[0][i])
        y_value.append(d[i]*eig_vec[1][i])
    plt.scatter(D[:,[0]],D[:,[1]],label='Genrerated data points')   
    plt.title(f"Projected values on {i+1}st eigonvector")
    plt.quiver(*origin, *eig_vec1, color=['r'], scale=5)
    plt.quiver(*origin, *eig_vec2, color=['b'], scale=10)
    plt.scatter(x_value,y_value,label='projed data points') 
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

# Part-d
# reconstructe the data
pca=PCA(n_components=2)
projected_data=pca.fit_transform(D)
reconsturcte_data=pca.inverse_transform(projected_data)

# reconstructe error
rec_error=np.linalg.norm((D-reconsturcte_data),None)
print("Reconstruction error")
print('%.2f'%rec_error)



