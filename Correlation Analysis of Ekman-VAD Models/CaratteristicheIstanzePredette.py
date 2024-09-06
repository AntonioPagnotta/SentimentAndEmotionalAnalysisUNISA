import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import spacy
import pandas as pd

def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

def count_syllables_in_phrase(phrase):
    doc = nlp(phrase)
    total_syllables = sum(count_syllables(token.text) for token in doc)
    return total_syllables

def get_noun_chunk_lengths(doc):
    return [len(chunk.text) for chunk in doc.noun_chunks]

nlp = spacy.load("en_core_web_sm")

data = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier.csv')

data_neutral = data.loc[data['predicted_emotion'] == 'neutral']
data_joy = data.loc[data['predicted_emotion'] == 'joy']
data_surprise = data.loc[data['predicted_emotion'] == 'surprise']
data_sadness = data.loc[data['predicted_emotion'] == 'sadness']
data_anger = data.loc[data['predicted_emotion'] == 'anger']
data_fear = data.loc[data['predicted_emotion'] == 'fear']
data_disgust = data.loc[data['predicted_emotion'] == 'disgust']

data_neutral['number_of_verbs'] = 0
data_neutral['number_of_syllables'] = 0
data_neutral['phrase_length'] = 0

print("Emotion - Neutral")
for index, row in data_neutral.iterrows():
    doc = nlp(row['text'])
    data_neutral.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_neutral.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_neutral.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_neutral.head())


print("\nEmotion - Joy")
data_joy['number_of_verbs'] = 0
data_joy['number_of_syllables'] = 0
data_joy['phrase_length'] = 0

for index, row in data_joy.iterrows():
    doc = nlp(row['text'])
    data_joy.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_joy.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_joy.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_joy.head())

print("\nEmotion - Surprise")
data_surprise['number_of_verbs'] = 0
data_surprise['number_of_syllables'] = 0
data_surprise['phrase_length'] = 0

for index, row in data_surprise.iterrows():
    doc = nlp(row['text'])
    data_surprise.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_surprise.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_surprise.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_surprise.head())

print("\nEmotion - Sadness")
data_sadness['number_of_verbs'] = 0
data_sadness['number_of_syllables'] = 0
data_sadness['phrase_length'] = 0

for index, row in data_sadness.iterrows():
    doc = nlp(row['text'])
    data_sadness.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_sadness.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_sadness.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_sadness.head())

print("\nEmotion - Anger")
data_anger['number_of_verbs'] = 0
data_anger['number_of_syllables'] = 0
data_anger['phrase_length'] = 0

for index, row in data_anger.iterrows():
    doc = nlp(row['text'])
    data_anger.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_anger.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_anger.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_anger.head())

print("\nEmotion - Fear")
data_fear['number_of_verbs'] = 0
data_fear['number_of_syllables'] = 0
data_fear['phrase_length'] = 0

for index, row in data_fear.iterrows():
    doc = nlp(row['text'])
    data_fear.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_fear.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_fear.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_fear.head())


print("\nEmotion - Disgust")
data_fear['number_of_verbs'] = 0
data_fear['number_of_syllables'] = 0
data_fear['phrase_length'] = 0

for index, row in data_fear.iterrows():
    doc = nlp(row['text'])
    data_fear.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data_fear.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data_fear.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data_fear.head())


"""
data['number_of_verbs'] = 0
data['number_of_syllables'] = 0
data['phrase_length'] = 0

for index, row in data.iterrows():
    doc = nlp(row['text'])
    data.at[index, 'number_of_verbs'] = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
    data.at[index, 'number_of_syllables'] = count_syllables_in_phrase(row['text'])
    data.at[index, 'phrase_length'] = math.trunc(len(row['text'].strip()))
print(data.head())
data.to_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis.csv', index=False)
"""

