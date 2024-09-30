import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
from sklearn.model_selection import train_test_split



'''
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

df = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis.csv')
df['text'] = df['text'].astype(str)  # Ensure all values are strings
df['text'] = df['text'].fillna('')  # Fill NaN values with empty string

total_phrases = df['text'].tolist()
embeddings = model.encode(total_phrases)
embedding_array = np.array(embeddings)
print(embeddings)
print(embedding_array)


embs = pd.DataFrame(embedding_array)
embs.columns = ['emb_{}'.format(i) for i in range(len(embs.columns))]
print(embs)
result = pd.concat([df, embs], axis=1)
print(result)


result.to_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis_v0.2.csv', index=False)
'''






df = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis_v0.2.csv')


data = df
data_neutral = data.loc[data['predicted_emotion'] == 'neutral']
data_joy = data.loc[data['predicted_emotion'] == 'joy']
data_surprise = data.loc[data['predicted_emotion'] == 'surprise']
data_sadness = data.loc[data['predicted_emotion'] == 'sadness']
data_anger = data.loc[data['predicted_emotion'] == 'anger']
data_fear = data.loc[data['predicted_emotion'] == 'fear']
data_disgust = data.loc[data['predicted_emotion'] == 'disgust']

primary_neutral, secondary_neutral = train_test_split(data_neutral, test_size=0.2, random_state=42)
primary_joy, secondary_joy = train_test_split(data_joy, test_size=0.2, random_state=42)
primary_surprise, secondary_surprise = train_test_split(data_surprise, test_size=0.2, random_state=42)
primary_sadness, secondary_sadness = train_test_split(data_sadness, test_size=0.2, random_state=42)
primary_anger, secondary_anger = train_test_split(data_anger, test_size=0.2, random_state=42)
primary_fear, secondary_fear = train_test_split(data_fear, test_size=0.2, random_state=42)
primary_disgust, secondary_disgust = train_test_split(data_disgust, test_size=0.2, random_state=42)

primary_data = pd.concat([primary_neutral, primary_joy, primary_surprise, primary_sadness, primary_anger, primary_fear, primary_disgust], axis=0).reset_index(drop=True)
secondary_data = pd.concat([secondary_neutral, secondary_joy, secondary_surprise, secondary_sadness, secondary_anger, secondary_fear, secondary_disgust], axis=0).reset_index(drop=True)

primary_data.to_csv('ekman_predictions_arpanghoshal_EkmanClassifier_PRIMARY_dataset_v0.2.csv', index=False)
secondary_data.to_csv('ekman_predictions_arpanghoshal_EkmanClassifier_SECONDARY_dataset_v0.2.csv', index=False)
