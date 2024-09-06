import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier.csv')
data_neutral = data.loc[data['predicted_emotion'] == 'neutral']
data_joy = data.loc[data['predicted_emotion'] == 'joy']
data_surprise = data.loc[data['predicted_emotion'] == 'surprise']
data_sadness = data.loc[data['predicted_emotion'] == 'sadness']
data_anger = data.loc[data['predicted_emotion'] == 'anger']
data_fear = data.loc[data['predicted_emotion'] == 'fear']
data_disgust = data.loc[data['predicted_emotion'] == 'disgust']



#ANGER
dominance = data_anger['D']
valence = data_anger['V']
arousal = data_anger['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.839], [1.969], [3.278], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Anger')
plt.show()



#DISGUST
dominance = data_disgust['D']
valence = data_disgust['V']
arousal = data_disgust['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.295], [1.680], [2.887], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Disgust')
plt.show()



#FEAR
dominance = data_fear['D']
valence = data_fear['V']
arousal = data_fear['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.720], [1.612], [1.969], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Fear')
plt.show()



#JOY
dominance = data_joy['D']
valence = data_joy['V']
arousal = data_joy['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.516], [3.992], [3.295], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Joy')
plt.show()



#NEUTRAL
dominance = data_neutral['D']
valence = data_neutral['V']
arousal = data_neutral['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Neutral')
plt.show()



#SADNESS
dominance = data_sadness['D']
valence = data_sadness['V']
arousal = data_sadness['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.159], [1.629], [2.139], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Sadness')
plt.show()



#SURPRISE
dominance = data_surprise['D']
valence = data_surprise['V']
arousal = data_surprise['A']

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, alpha=0.2)
ax.scatter([3.839], [3.380], [2.479], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Surprise')
plt.show()