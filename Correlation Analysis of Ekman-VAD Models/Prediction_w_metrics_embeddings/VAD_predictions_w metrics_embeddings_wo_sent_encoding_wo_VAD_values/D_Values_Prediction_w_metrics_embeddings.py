import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


primary_dataset = pd.read_csv('../../ekman_predictions_arpanghoshal_EkmanClassifier_PRIMARY_dataset_v0.2.csv')
secondary_dataset = pd.read_csv('../../ekman_predictions_arpanghoshal_EkmanClassifier_SECONDARY_dataset_v0.2.csv')

primary_dataset.info()
secondary_dataset.info()

primary_dataset = primary_dataset.dropna()
secondary_dataset = secondary_dataset.dropna()

X_train = primary_dataset.drop(['V'], axis=1)
X_train = X_train.drop(['A'], axis=1)
X_train = X_train.drop(['D'], axis=1).drop(['text'], axis=1)
X_train = X_train.drop(['predicted_emotion'], axis=1)
Y_train = primary_dataset['D']

X_test = secondary_dataset.drop(['V'], axis=1)
X_test = X_test.drop(['A'], axis=1)
X_test = X_test.drop(['D'], axis=1).drop(['text'], axis=1)
X_test = X_test.drop(['predicted_emotion'], axis=1)
Y_test = secondary_dataset['D']


#LINEAR REGRESSION
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

'''
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title(" Actual vs Predicted D Value")
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
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("LINEAR REGRESSOR D Values Predictions")
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
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("LASSO REGRESSOR D Values Predictions")
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
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("LASSO REGRESSOR D Values Predictions")
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
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("RANDOM FOREST REGRESSOR D Values Predictions")
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
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("RANDOM FOREST REGRESSOR D Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()



#GRADIENT BOOSTING REGRESSION
grad_boost_model = GradientBoostingRegressor()
grad_boost_model.fit(X_train,Y_train)

'''
training_data_prediction = grad_boost_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("RANDOM FOREST REGRESSOR D Values Predictions")
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
plt.xlabel("Actual D Value")
plt.ylabel("Predicted D Value")
plt.title("GRADIENT BOOSTING REGRESSOR D Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()



'''
#METRICHE

LINEAR REGRESSOR
Mean Absolute Error (MAE): 0.1480426011078327
Mean Squared Error (MSE): 0.03800904151317127
Root Mean Squared Error (RMSE): 0.1949590765088183 
R^2: 0.1377321257565074

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.1571442130073945
Mean Squared Error (MSE): 0.044080437277662886
Root Mean Squared Error (RMSE): 0.2099534169230472 
R^2: -2.721298896224411e-06

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.14698410116387509
Mean Squared Error (MSE): 0.03854699423353832
Root Mean Squared Error (RMSE): 0.1963338845781296 
R^2: 0.12552820452703728

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.14642524594162273
Mean Squared Error (MSE): 0.03805412275462
Root Mean Squared Error (RMSE): 0.19507465943740618 
R^2: 0.1367094189297966
'''