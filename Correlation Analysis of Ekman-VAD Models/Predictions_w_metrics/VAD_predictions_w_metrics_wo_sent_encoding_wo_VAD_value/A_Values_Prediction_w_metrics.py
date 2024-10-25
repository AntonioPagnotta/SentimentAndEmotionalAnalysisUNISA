import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import ensemble



data = pd.read_csv('../../ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis.csv')
primary_dataset = pd.read_csv('../../ekman_predictions_arpanghoshal_EkmanClassifier_PRIMARY_dataset.csv')
secondary_dataset = pd.read_csv('../../ekman_predictions_arpanghoshal_EkmanClassifier_SECONDARY_dataset.csv')


data = data.drop(['text'], axis=1)
primary_dataset = primary_dataset.drop(['text'], axis=1)
secondary_dataset = secondary_dataset.drop(['text'], axis=1)


'''
Emotion encoding:
    "0": "anger",
    "1": "disgust",
    "2": "fear",
    "3": "joy",
    "4": "neutral",
    "5": "sadness",
    "6": "surprise"
'''


data.info()
primary_dataset.info()
secondary_dataset.info()

data = data.dropna()
primary_dataset = primary_dataset.dropna()
secondary_dataset = secondary_dataset.dropna()

X_train = primary_dataset.drop(['V'], axis=1)
X_train = X_train.drop(['A'], axis=1)
X_train = X_train.drop(['D'], axis=1)
X_train = X_train.drop(['predicted_emotion'], axis=1)
Y_train = primary_dataset['A']

X_test = secondary_dataset.drop(['V'], axis=1)
X_test = X_test.drop(['A'], axis=1)
X_test = X_test.drop(['D'], axis=1)
X_test = X_test.drop(['predicted_emotion'], axis=1)
Y_test = secondary_dataset['A']


#LINEAR REGRESSION
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

'''
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title(" Actual vs Predicted A Value")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()
'''

test_data_prediction = lin_reg_model.predict(X_test)
print("\nLINEAR REGRESSOR")
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(Y_test, test_data_prediction))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test, test_data_prediction))
print('\033[32mRoot Mean Squared Error (RMSE):', metrics.mean_squared_error(Y_test, test_data_prediction, squared=False), '\033[0m')
print('R^2:', metrics.r2_score(Y_test, test_data_prediction))

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title("LINEAR REGRESSOR A Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()



#LASSO REGRESSOR
lass_reg_model = Lasso()
lass_reg_model.fit(X_train,Y_train)

'''
training_data_prediction = lass_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title(" Actual vs Predicted A Value")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()
'''

test_data_prediction = lass_reg_model.predict(X_test)
print("\nLASSO REGRESSOR")
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(Y_test, test_data_prediction))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test, test_data_prediction))
print('\033[32mRoot Mean Squared Error (RMSE):', metrics.mean_squared_error(Y_test, test_data_prediction, squared=False), '\033[0m')
print('R^2:', metrics.r2_score(Y_test, test_data_prediction))

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title("LASSO REGRESSOR A Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()



#RANDOM FOREST REGRESSOR
rf=RandomForestRegressor()
rf.fit(X_train,Y_train)

'''
training_data_prediction=rf.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title(" Actual vs Predicted A Value")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()
'''

test_data_prediction=rf.predict(X_test)
print("\nRANDOM FOREST REGRESSOR")
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(Y_test, test_data_prediction))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test, test_data_prediction))
print('\033[32mRoot Mean Squared Error (RMSE):', metrics.mean_squared_error(Y_test, test_data_prediction, squared=False), '\033[0m')
print('R^2:', metrics.r2_score(Y_test, test_data_prediction))

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title("RANDOM FOREST REGRESSOR A Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()



#GRADIENT BOOSTING REGRESSION
grad_boost_model = ensemble.GradientBoostingRegressor()
grad_boost_model.fit(X_train,Y_train)

'''
training_data_prediction = grad_boost_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title(" Actual vs Predicted A Value")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()
'''

test_data_prediction = grad_boost_model.predict(X_test)
print("\nGRADIENT BOOSTING REGRESSOR")
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(Y_test, test_data_prediction))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test, test_data_prediction))
print('\033[32mRoot Mean Squared Error (RMSE):', metrics.mean_squared_error(Y_test, test_data_prediction, squared=False), '\033[0m')
print('R^2:', metrics.r2_score(Y_test, test_data_prediction))

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual A Value")
plt.ylabel("Predicted A Value")
plt.title("GRADIENT BOOSTING REGRESSOR A Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()


'''
#METRICHE

LINEAR REGRESSOR
Mean Absolute Error (MAE): 0.19510348390282173
Mean Squared Error (MSE): 0.06776584332549843
Root Mean Squared Error (RMSE): 0.2603187341039796 
R^2: 0.01370872754663599

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.19698683793927288
Mean Squared Error (MSE): 0.06878651105414557
Root Mean Squared Error (RMSE): 0.2622718266496529 
R^2: -0.001146480083616952

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.20413390914269916
Mean Squared Error (MSE): 0.07348710845417968
Root Mean Squared Error (RMSE): 0.27108505760034 
R^2: -0.06956086059537192

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.19485207187281425
Mean Squared Error (MSE): 0.06740095516177165
Root Mean Squared Error (RMSE): 0.2596169392812643 
R^2: 0.01901945628023427
'''