import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv('p_emotion.csv')

data = data[['V', 'A', 'D', 'predicted_emotion']]
data.replace({'predicted_emotion':{'joy':0,'anticipation':1,'surprise':2,'sadness':3}},inplace=True)


X = data.drop(['predicted_emotion'], axis=1)

Y = data['predicted_emotion'].values.reshape(-1,1).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


classifier = SVC(kernel='linear', C=1)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
conf_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("\nConfusion Matrix:")
print(conf_mat)