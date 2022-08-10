""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Part-A (Lab-5)

# import module from python library
import pandas as pd 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix , accuracy_score

# read the csv file of spilting data of train and test set using pandas library
train_data=pd.read_csv('SteelPlateFaults-train.csv') 
test_data=pd.read_csv('SteelPlateFaults-test.csv') 

# remove the some data column using drop function
train_data=train_data.drop(columns=[train_data.columns[0],'X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'])
test_data=test_data.drop(columns=[test_data.columns[0],'X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'])

# Classifing the train data basis on classes (class 1 or 0)
train_0=train_data[train_data['Class']==0].drop(columns=['Class'])
train_1=train_data[train_data['Class']==1].drop(columns=['Class'])

# Classifing the TEST data basis on classes (class 1 or 0)
test_0=test_data[test_data['Class']==0].drop(columns=['Class'])
test_1=test_data[test_data['Class']==1].drop(columns=['Class'])

# assume the value of maximum accurancy and corrosponding Q-value
Max_acc=0
Max_acc_Q=2
# Put the all Q-value in a list for GaussianMixture model predection
Q_list=[2,4,8,16]
for Q in Q_list:
    # make the GaussianMixture model for class 0 and class 1 using sklearn library function 
    gmm_0 = GaussianMixture(n_components=Q,covariance_type='full',reg_covar=1e-4)
    gmm_1 = GaussianMixture(n_components=Q,covariance_type='full',reg_covar=1e-4)
    gmm_0.fit(train_0)
    gmm_1.fit(train_1)
    # find the probability of sample class of all the test data using score_samples function
    y_0=gmm_0.score_samples(test_data.iloc[:,:23])
    y_1=gmm_1.score_samples(test_data.iloc[:,:23])
    # Predect the class of test data using for loop
    pred=[]
    for i in range(len(y_1)):
        if y_0[i]>y_1[i]:
            pred.append(0)
        else:
            pred.append(1)
    # find the confusion matrix oand accuracy_score of test data for different Q-value 
    conf_matrix = confusion_matrix(test_data['Class'],pred)
    predicted_acc=  accuracy_score(test_data['Class'],pred)

    print('Accuracy for the data predicted for ',Q,' components is :', round(predicted_acc,3))
    print("Confusion matrix for Q =",Q," is ---> ")
    print(conf_matrix)
    if(predicted_acc>Max_acc):
        Max_acc=predicted_acc
        Max_acc_Q=Q
    print("------")
# print the highest accuracy and corrosponding Q-value    
print("Highest accuracy of the GMM model is: ",Max_acc," for Q: ",Max_acc_Q)
print("-----------------------------------------------------------------------------------")