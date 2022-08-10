""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for question 4 (Lab-1)

# import module from library
import pandas as pd
import matplotlib.pyplot as plt

# reading the csv file using pandas
data=pd.read_csv("C:\our data\college related\DS 3 COURSE\ALL ABOUT LAB\LAB-1\lab assignment 1\pima-indians-diabetes.csv")

# Plot the histogram of pregs attribute using matlpotlib.pyplot library
plt.title('Histogram plot for attribute preg',fontsize=12)
plt.hist(data['pregs'])
plt.show()

# Plot the histogram of pregs attribute using matlpotlib.pyplot library
plt.title('Histogram plot for attribute skin',fontsize=12)
plt.hist(data['skin'])
plt.show()