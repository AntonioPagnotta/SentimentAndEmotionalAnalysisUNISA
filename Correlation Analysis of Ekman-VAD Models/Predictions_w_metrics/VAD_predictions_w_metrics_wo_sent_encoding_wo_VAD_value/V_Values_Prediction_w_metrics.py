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
Y_train = primary_dataset['V']

X_test = secondary_dataset.drop(['V'], axis=1)
X_test = X_test.drop(['A'], axis=1)
X_test = X_test.drop(['D'], axis=1)
X_test = X_test.drop(['predicted_emotion'], axis=1)
Y_test = secondary_dataset['V']



#LINEAR REGRESSOR
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
error_score = metrics.r2_score(Y_test, test_data_prediction)
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
error_score = metrics.r2_score(Y_test, test_data_prediction)
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



#RANDOM_FOREST_REGRESSOR
rf=RandomForestRegressor() #n_estimators = 1000,max_depth=10,random_state = 0
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
error_score = metrics.r2_score(Y_test, test_data_prediction)
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
grad_boost_model = ensemble.GradientBoostingRegressor()
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
METRICHE

LINEAR REGRESSOR
Mean Absolute Error (MAE): 0.2511435393850646
Mean Squared Error (MSE): 0.12306593732901384
Root Mean Squared Error (RMSE): 0.3508075502736705 
R^2: -0.0009736975865317632

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.24965911089742837
Mean Squared Error (MSE): 0.12297292616288563
Root Mean Squared Error (RMSE): 0.35067495799227755 
R^2: -0.0002171784968727497

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.2626159989927064
Mean Squared Error (MSE): 0.13007661471663376
Root Mean Squared Error (RMSE): 0.3606613573931005 
R^2: -0.05799600464872934

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.2508229621745022
Mean Squared Error (MSE): 0.12317460356625899
Root Mean Squared Error (RMSE): 0.350962396228227 
R^2: -0.001857549346481413

'''
