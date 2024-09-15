import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import textstat

def count_unique_words(sentence):
    sentence = ''.join(c for c in sentence if c.isalnum() or c.isspace()).lower()
    words = sentence.split()
    unique_words = set(words)
    return len(unique_words)


def calculate_mean_std(sentence):
    sentence = sentence.rstrip('.!?')
    words = sentence.split()
    lengths = [len(word) for word in words]
    media = np.mean(lengths)
    dev_std = np.std(lengths)
    return media, dev_std



textstat.set_lang("en_US")

data = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier.csv')

data_neutral = data.loc[data['predicted_emotion'] == 'neutral']
data_joy = data.loc[data['predicted_emotion'] == 'joy']
data_surprise = data.loc[data['predicted_emotion'] == 'surprise']
data_sadness = data.loc[data['predicted_emotion'] == 'sadness']
data_anger = data.loc[data['predicted_emotion'] == 'anger']
data_fear = data.loc[data['predicted_emotion'] == 'fear']
data_disgust = data.loc[data['predicted_emotion'] == 'disgust']



#NEUTRAL
print("Emotion - Neutral")
data_neutral['word_count'] = data_neutral['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_neutral['syllable_count'] = data_neutral['text'].apply(lambda x: textstat.syllable_count(x))
data_neutral['letter_count'] = data_neutral['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_neutral['unique_words'] = data_neutral['text'].apply(lambda x: count_unique_words(x))
results = data_neutral['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_neutral['mean_word_length'] = means
data_neutral['std_dev_word_length'] = std_devs

data_neutral['flesch_reading_ease']= data_neutral['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_neutral['flesch_kincaid_grade']= data_neutral['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_neutral['gunning_fog']= data_neutral['text'].apply(lambda x: textstat.gunning_fog(x))
data_neutral['automated_readability_index']= data_neutral['text'].apply(lambda x: textstat.automated_readability_index(x))
data_neutral['coleman_liau_index']= data_neutral['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_neutral['linsear_write_formula']= data_neutral['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_neutral['dale_chall_readability_score']= data_neutral['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_neutral['text_standard']= data_neutral['text'].apply(lambda x: textstat.text_standard(x))
data_neutral['mcalpine_eflaw']= data_neutral['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_neutral['reading_time']= data_neutral['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_neutral.head())


#JOY
print("Emotion - Joy")
data_joy['word_count'] = data_joy['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_joy['syllable_count'] = data_joy['text'].apply(lambda x: textstat.syllable_count(x))
data_joy['letter_count'] = data_joy['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_joy['unique_words'] = data_joy['text'].apply(lambda x: count_unique_words(x))
results = data_joy['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_joy['mean_word_length'] = means
data_joy['std_dev_word_length'] = std_devs

data_joy['flesch_reading_ease']= data_joy['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_joy['flesch_kincaid_grade']= data_joy['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_joy['gunning_fog']= data_joy['text'].apply(lambda x: textstat.gunning_fog(x))
data_joy['automated_readability_index']= data_joy['text'].apply(lambda x: textstat.automated_readability_index(x))
data_joy['coleman_liau_index']= data_joy['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_joy['linsear_write_formula']= data_joy['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_joy['dale_chall_readability_score']= data_joy['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_joy['text_standard']= data_joy['text'].apply(lambda x: textstat.text_standard(x))
data_joy['mcalpine_eflaw']= data_joy['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_joy['reading_time']= data_joy['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_joy.head())

#SURPRISE
print("Emotion - Surprise")
data_surprise['word_count'] = data_surprise['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_surprise['syllable_count'] = data_surprise['text'].apply(lambda x: textstat.syllable_count(x))
data_surprise['letter_count'] = data_surprise['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_surprise['unique_words'] = data_surprise['text'].apply(lambda x: count_unique_words(x))
results = data_surprise['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_surprise['mean_word_length'] = means
data_surprise['std_dev_word_length'] = std_devs

data_surprise['flesch_reading_ease']= data_surprise['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_surprise['flesch_kincaid_grade']= data_surprise['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_surprise['gunning_fog']= data_surprise['text'].apply(lambda x: textstat.gunning_fog(x))
data_surprise['automated_readability_index']= data_surprise['text'].apply(lambda x: textstat.automated_readability_index(x))
data_surprise['coleman_liau_index']= data_surprise['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_surprise['linsear_write_formula']= data_surprise['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_surprise['dale_chall_readability_score']= data_surprise['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_surprise['text_standard']= data_surprise['text'].apply(lambda x: textstat.text_standard(x))
data_surprise['mcalpine_eflaw']= data_surprise['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_surprise['reading_time']= data_surprise['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_surprise.head())



#SADNESS
print("Emotion - Sadness")
data_sadness['word_count'] = data_sadness['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_sadness['syllable_count'] = data_sadness['text'].apply(lambda x: textstat.syllable_count(x))
data_sadness['letter_count'] = data_sadness['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_sadness['unique_words'] = data_sadness['text'].apply(lambda x: count_unique_words(x))
results = data_sadness['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_sadness['mean_word_length'] = means
data_sadness['std_dev_word_length'] = std_devs

data_sadness['flesch_reading_ease']= data_sadness['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_sadness['flesch_kincaid_grade']= data_sadness['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_sadness['gunning_fog']= data_sadness['text'].apply(lambda x: textstat.gunning_fog(x))
data_sadness['automated_readability_index']= data_sadness['text'].apply(lambda x: textstat.automated_readability_index(x))
data_sadness['coleman_liau_index']= data_sadness['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_sadness['linsear_write_formula']= data_sadness['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_sadness['dale_chall_readability_score']= data_sadness['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_sadness['text_standard']= data_sadness['text'].apply(lambda x: textstat.text_standard(x))
data_sadness['mcalpine_eflaw']= data_sadness['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_sadness['reading_time']= data_sadness['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_sadness.head())




#ANGER
print("Emotion - Anger")
data_anger['word_count'] = data_anger['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_anger['syllable_count'] = data_anger['text'].apply(lambda x: textstat.syllable_count(x))
data_anger['letter_count'] = data_anger['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_anger['unique_words'] = data_anger['text'].apply(lambda x: count_unique_words(x))
results = data_anger['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_anger['mean_word_length'] = means
data_anger['std_dev_word_length'] = std_devs

data_anger['flesch_reading_ease']= data_anger['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_anger['flesch_kincaid_grade']= data_anger['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_anger['gunning_fog']= data_anger['text'].apply(lambda x: textstat.gunning_fog(x))
data_anger['automated_readability_index']= data_anger['text'].apply(lambda x: textstat.automated_readability_index(x))
data_anger['coleman_liau_index']= data_anger['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_anger['linsear_write_formula']= data_anger['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_anger['dale_chall_readability_score']= data_anger['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_anger['text_standard']= data_anger['text'].apply(lambda x: textstat.text_standard(x))
data_anger['mcalpine_eflaw']= data_anger['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_anger['reading_time']= data_anger['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_anger.head())



#FEAR
print("Emotion - Fear")
data_fear['word_count'] = data_fear['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_fear['syllable_count'] = data_fear['text'].apply(lambda x: textstat.syllable_count(x))
data_fear['letter_count'] = data_fear['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_fear['unique_words'] = data_fear['text'].apply(lambda x: count_unique_words(x))
results = data_fear['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_fear['mean_word_length'] = means
data_fear['std_dev_word_length'] = std_devs

data_fear['flesch_reading_ease']= data_fear['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_fear['flesch_kincaid_grade']= data_fear['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_fear['gunning_fog']= data_fear['text'].apply(lambda x: textstat.gunning_fog(x))
data_fear['automated_readability_index']= data_fear['text'].apply(lambda x: textstat.automated_readability_index(x))
data_fear['coleman_liau_index']= data_fear['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_fear['linsear_write_formula']= data_fear['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_fear['dale_chall_readability_score']= data_fear['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_fear['text_standard']= data_fear['text'].apply(lambda x: textstat.text_standard(x))
data_fear['mcalpine_eflaw']= data_fear['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_fear['reading_time']= data_fear['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_fear.head())




#DISGUST
print("Emotion - Disgust")
data_disgust['word_count'] = data_disgust['text'].apply(lambda x: textstat.lexicon_count(x, removepunct=True))
data_disgust['syllable_count'] = data_disgust['text'].apply(lambda x: textstat.syllable_count(x))
data_disgust['letter_count'] = data_disgust['text'].apply(lambda x: textstat.letter_count(x, ignore_spaces=True))
data_disgust['unique_words'] = data_disgust['text'].apply(lambda x: count_unique_words(x))
results = data_disgust['text'].apply(lambda x: calculate_mean_std(x))
means, std_devs = zip(*results)
data_disgust['mean_word_length'] = means
data_disgust['std_dev_word_length'] = std_devs

data_disgust['flesch_reading_ease']= data_disgust['text'].apply(lambda x: textstat.flesch_reading_ease(x))
data_disgust['flesch_kincaid_grade']= data_disgust['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
data_disgust['gunning_fog']= data_disgust['text'].apply(lambda x: textstat.gunning_fog(x))
data_disgust['automated_readability_index']= data_disgust['text'].apply(lambda x: textstat.automated_readability_index(x))
data_disgust['coleman_liau_index']= data_disgust['text'].apply(lambda x: textstat.coleman_liau_index(x))
data_disgust['linsear_write_formula']= data_disgust['text'].apply(lambda x: textstat.linsear_write_formula(x))
data_disgust['dale_chall_readability_score']= data_disgust['text'].apply(lambda x: textstat.dale_chall_readability_score(x))
data_disgust['text_standard']= data_disgust['text'].apply(lambda x: textstat.text_standard(x))
data_disgust['mcalpine_eflaw']= data_disgust['text'].apply(lambda x: textstat.mcalpine_eflaw(x))
data_disgust['reading_time']= data_disgust['text'].apply(lambda x: textstat.reading_time(x, ms_per_char=14.69))
print(data_disgust.head())



dataframes = [data_neutral, data_joy, data_surprise, data_sadness, data_anger, data_fear, data_disgust]
emotion_names = ['neutral', 'joy', 'surprise', 'sadness', 'anger', 'fear', 'disgust']
for df, emotion in zip(dataframes, emotion_names):
    df['predicted_emotion'] = emotion
combined_data = pd.concat(dataframes, axis=0)
combined_data.reset_index(drop=True, inplace=True)
print(combined_data.head())

combined_data.to_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis.csv', index=False)
