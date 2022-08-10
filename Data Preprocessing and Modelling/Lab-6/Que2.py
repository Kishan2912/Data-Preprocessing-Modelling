""" Name- Kishan Sharma 
    Roll No - B20294
    Mob no - 8000543233 """
# Python code for Question-2 (Lab-6)

# import python libraries
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

# Part-A
# spilting the covid cases data into train (65%) and test data (35%)
series = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
train_size = 0.65
X = series.values
train, test = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]

# train AR model and predict using the coefficients
# use lagged value =5
window = 5 
model = AR(train, lags=window)
# fit/train the model
model_fit = model.fit()
# Get the coefficients of AR model (w1,w2,w3......)
coeff = model_fit.params 
print("Part-A ---> ")
print("From AR model Coefficients (w1,w2,w3......) are ",coeff)
print("-----------------------")

# using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
# List to hold the predictions, 1 step at a time
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coeff[0]
    for d in range(window):
        yhat += coeff[d+1] * lag[window-d-1]
    obs = test[t]
    # Append predictions to compute RMSE later
    predictions.append(yhat) 
    # Append actual test value to history, to be used in next step
    history.append(obs)


# Part-B
# part-(i)
# scatter plot between actual and predicted values of covid cases
plt.scatter(test,predictions , color="red")
plt.xlabel('Actual cases')
plt.ylabel('Predicted cases')
plt.title('Scatter plot between actual and predicted cases')
plt.show()

# part-(ii)
# line plot showing actual and predicted test values of covid cases
x=[i for i in range(len(test))]
plt.plot(x,test, label='Actual cases',color="red")
plt.plot(x,predictions , label='Predicted cases')
plt.legend()
plt.title('Line plot of actual and pridect value')
plt.show()

# part-(iii)
print("Part-B (iii) ---> ")
# Find the RMSE (%) value between actual and predicted test data
rmse=mean_squared_error(test, predictions,squared=False)
print("RMSE (%) value between actual and predicted test data is : ",rmse*100/(sum(test)/len(test)),"%")

# find the MAPE value between actual and predicted test data
mape=mean_absolute_percentage_error(test, predictions)
print("MAPE value between actual and predicted test data is : ",mape)
print("-------------------------------------------------")
