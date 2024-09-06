import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

data = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier.csv')

data_neutral = data.loc[data['predicted_emotion'] == 'neutral']
data_joy = data.loc[data['predicted_emotion'] == 'joy']
data_surprise = data.loc[data['predicted_emotion'] == 'surprise']
data_sadness = data.loc[data['predicted_emotion'] == 'sadness']
data_anger = data.loc[data['predicted_emotion'] == 'anger']
data_fear = data.loc[data['predicted_emotion'] == 'fear']
data_disgust = data.loc[data['predicted_emotion'] == 'disgust']

X = data_neutral[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_neutral['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_neutral['A'], data_neutral['V'], data_neutral['D'], c=data_neutral['HC_cluster'])
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Neutral Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()



X = data_joy[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_joy['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_joy['A'], data_joy['V'], data_joy['D'], c=data_joy['HC_cluster'])
ax.scatter([3.516], [3.992], [3.295], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Joy Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()



X = data_surprise[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_surprise['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_surprise['A'], data_surprise['V'], data_surprise['D'], c=data_surprise['HC_cluster'])
ax.scatter([3.839], [3.380], [2.479], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Surprise Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()



X = data_sadness[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_sadness['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_sadness['A'], data_sadness['V'], data_sadness['D'], c=data_sadness['HC_cluster'])
ax.scatter([3.159], [1.629], [2.139], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Sadness Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()



X = data_anger[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_anger['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_anger['A'], data_anger['V'], data_anger['D'], c=data_anger['HC_cluster'])
ax.scatter([3.839], [1.969], [3.278], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Anger Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()



X = data_fear[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_fear['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_fear['A'], data_fear['V'], data_fear['D'], c=data_fear['HC_cluster'])
ax.scatter([3.720], [1.612], [1.969], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Fear Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()



X = data_disgust[['D', 'V', 'A']]
dist_matrix = pdist(X, metric='euclidean')
Z = linkage(dist_matrix, method='ward')
n_clusters = 3
clusters_ward = fcluster(Z, n_clusters, criterion='maxclust')
data_disgust['HC_cluster'] = clusters_ward

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
scatter = ax.scatter(data_disgust['A'], data_disgust['V'], data_disgust['D'], c=data_disgust['HC_cluster'])
ax.scatter([3.295], [1.680], [2.887], color='red', s=40, marker='o', alpha=0.4)
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title(f'Disgust Emotion Hierarchical Clustering con {n_clusters} cluster')
plt.show()