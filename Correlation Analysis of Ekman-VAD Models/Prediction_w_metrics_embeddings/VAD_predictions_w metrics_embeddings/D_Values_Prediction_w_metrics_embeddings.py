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

primary_dataset.replace({'predicted_emotion':{'anger':0,'disgust':1,'fear':2,'joy':3,'neutral':4,'sadness':5,'surprise':6}},inplace=True)
secondary_dataset.replace({'predicted_emotion':{'anger':0,'disgust':1,'fear':2,'joy':3,'neutral':4,'sadness':5,'surprise':6}},inplace=True)

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

primary_dataset.info()
secondary_dataset.info()

primary_dataset = primary_dataset.dropna()
secondary_dataset = secondary_dataset.dropna()

X_train = primary_dataset.drop(['V'], axis=1).drop(['text'], axis=1)
Y_train = primary_dataset['V']

X_test = secondary_dataset.drop(['V'], axis=1).drop(['text'], axis=1)
Y_test = secondary_dataset['V']


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
Mean Absolute Error (MAE): 0.18092736621519268
Mean Squared Error (MSE): 0.059611516263669465
Root Mean Squared Error (RMSE): 0.2441546974024245 
R^2: 0.5151415481144896

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.24965911089742837
Mean Squared Error (MSE): 0.12297292616288563
Root Mean Squared Error (RMSE): 0.35067495799227755 
R^2: -0.0002171784968727497

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.1777807372204095
Mean Squared Error (MSE): 0.0623745106194169
Root Mean Squared Error (RMSE): 0.24974889513152385 
R^2: 0.49266835417708643

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.17620046104121792
Mean Squared Error (MSE): 0.059272482855059164
Root Mean Squared Error (RMSE): 0.24345940699644195 
R^2: 0.5178991228909668


'''