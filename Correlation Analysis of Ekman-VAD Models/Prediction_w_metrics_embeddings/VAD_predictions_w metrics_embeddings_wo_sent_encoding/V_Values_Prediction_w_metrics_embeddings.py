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

X_train = primary_dataset.drop(['V'], axis=1).drop(['text'], axis=1)
X_train = X_train.drop(['predicted_emotion'], axis=1)
Y_train = primary_dataset['V']

X_test = secondary_dataset.drop(['V'], axis=1).drop(['text'], axis=1)
X_test = X_test.drop(['predicted_emotion'], axis=1)
Y_test = secondary_dataset['V']


#LINEAR REGRESSION
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)

'''
training_data_prediction = lin_reg_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title(" Actual vs Predicted V Value")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title("LINEAR REGRESSOR V Values Predictions")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title(" Actual vs Predicted V Value")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title("LASSO REGRESSOR V Values Predictions")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title(" Actual vs Predicted V Value")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title("RANDOM FOREST REGRESSOR V Values Predictions")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title(" Actual vs Predicted V Value")
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
plt.xlabel("Actual V Value")
plt.ylabel("Predicted V Value")
plt.title("GRADIENT BOOSTING REGRESSOR V Values Predictions")
plt.xlim(1, 5)
plt.ylim(1, 5)
plt.show()



'''
#METRICHE

LINEAR REGRESSOR
Mean Absolute Error (MAE): 0.18107477480968923
Mean Squared Error (MSE): 0.05966348125227911
Root Mean Squared Error (RMSE): 0.244261092383292 
R^2: 0.514718883745107

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.24965911089742837
Mean Squared Error (MSE): 0.12297292616288563
Root Mean Squared Error (RMSE): 0.35067495799227755 
R^2: -0.0002171784968727497

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.1814077623303542
Mean Squared Error (MSE): 0.06433466352587444
Root Mean Squared Error (RMSE): 0.253642787253796 
R^2: 0.4767251813934894

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.1796015368250854
Mean Squared Error (MSE): 0.062105912657862715
Root Mean Squared Error (RMSE): 0.24921057894451976 
R^2: 0.49485303257450775

'''