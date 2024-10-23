import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


primary_dataset = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier_PRIMARY_dataset_v0.2.csv')
secondary_dataset = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier_SECONDARY_dataset_v0.2.csv')

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

X_train = primary_dataset.drop(['A'], axis=1).drop(['text'], axis=1).drop(['word_count'], axis=1).drop(['syllable_count'], axis=1).drop(['letter_count'], axis=1).drop(['unique_words'], axis=1).drop(['mean_word_length'], axis=1).drop(['std_dev_word_length'], axis=1).drop(['flesch_reading_ease'], axis=1).drop(['flesch_kincaid_grade'], axis=1).drop(['gunning_fog'], axis=1).drop(['automated_readability_index'], axis=1).drop(['coleman_liau_index'], axis=1).drop(['linsear_write_formula'], axis=1).drop(['dale_chall_readability_score'], axis=1).drop(['mcalpine_eflaw'], axis=1).drop(['reading_time'], axis=1)
X_train = X_train.drop(['predicted_emotion'], axis=1)
Y_train = primary_dataset['A']

X_test = secondary_dataset.drop(['A'], axis=1).drop(['text'], axis=1).drop(['word_count'], axis=1).drop(['syllable_count'], axis=1).drop(['letter_count'], axis=1).drop(['unique_words'], axis=1).drop(['mean_word_length'], axis=1).drop(['std_dev_word_length'], axis=1).drop(['flesch_reading_ease'], axis=1).drop(['flesch_kincaid_grade'], axis=1).drop(['gunning_fog'], axis=1).drop(['automated_readability_index'], axis=1).drop(['coleman_liau_index'], axis=1).drop(['linsear_write_formula'], axis=1).drop(['dale_chall_readability_score'], axis=1).drop(['mcalpine_eflaw'], axis=1).drop(['reading_time'], axis=1)
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
grad_boost_model = GradientBoostingRegressor()
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
Mean Absolute Error (MAE): 0.17050541429271918
Mean Squared Error (MSE): 0.048371224917990556
Root Mean Squared Error (RMSE): 0.21993459236325366 
R^2: 0.29598578526740593

LASSO REGRESSOR
Mean Absolute Error (MAE): 0.19698683793927288
Mean Squared Error (MSE): 0.06878651105414557
Root Mean Squared Error (RMSE): 0.2622718266496529 
R^2: -0.001146480083616952

RANDOM FOREST REGRESSOR
Mean Absolute Error (MAE): 0.1590289946564525
Mean Squared Error (MSE): 0.04357181385709752
Root Mean Squared Error (RMSE): 0.2087386256951442 
R^2: 0.36583833944485555

GRADIENT BOOSTING REGRESSOR
Mean Absolute Error (MAE): 0.15582117461767503
Mean Squared Error (MSE): 0.04106488014311282
Root Mean Squared Error (RMSE): 0.20264471407641704 
R^2: 0.4023252585383844

'''