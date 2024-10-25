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


data.replace({'predicted_emotion':{'anger':0,'disgust':1,'fear':2,'joy':3,'neutral':4,'sadness':5,'surprise':6}},inplace=True)
primary_dataset.replace({'predicted_emotion':{'anger':0,'disgust':1,'fear':2,'joy':3,'neutral':4,'sadness':5,'surprise':6}},inplace=True)
secondary_dataset.replace({'predicted_emotion':{'anger':0,'disgust':1,'fear':2,'joy':3,'neutral':4,'sadness':5,'surprise':6}},inplace=True)


data.info()
primary_dataset.info()
secondary_dataset.info()

data = data.dropna()
primary_dataset = primary_dataset.dropna()
secondary_dataset = secondary_dataset.dropna()

X_train = primary_dataset.drop(['V'], axis=1)
X_train = X_train.drop(['A'], axis=1)
X_train = X_train.drop(['D'], axis=1)
Y_train = primary_dataset['D']

X_test = secondary_dataset.drop(['V'], axis=1)
X_test = X_test.drop(['A'], axis=1)
X_test = X_test.drop(['D'], axis=1)
Y_test = secondary_dataset['D']


#LINEAR REGRESSOR
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
plt.title(" Actual vs Predicted D Value")
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
plt.title(" Actual vs Predicted D Value")
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
grad_boost_model = ensemble.GradientBoostingRegressor()
grad_boost_model.fit(X_train,Y_train)

'''
training_data_prediction = grad_boost_model.predict(X_train)
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
Mean Absolute Error (MAE): 0.1547017014360949
Mean Squared Error (MSE): 0.043131253659841585
Root Mean Squared Error (RMSE): 0.2076806530706257 
R^2: 0.021530327360641976

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.1571442130073945
Mean Squared Error (MSE): 0.044080437277662886
Root Mean Squared Error (RMSE): 0.2099534169230472 
R^2: -2.721298896224411e-06

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.1590455101340336
Mean Squared Error (MSE): 0.04491493443613853
Root Mean Squared Error (RMSE): 0.21193143805518455 
R^2: -0.01893400875721518

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.15237239163619962
Mean Squared Error (MSE): 0.04213831111873696
Root Mean Squared Error (RMSE): 0.2052761825413191 
R^2: 0.044056084919340766
'''