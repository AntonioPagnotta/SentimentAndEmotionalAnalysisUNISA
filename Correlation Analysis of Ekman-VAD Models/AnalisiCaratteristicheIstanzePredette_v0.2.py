import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis.csv')
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


#NEUTRAL
data_neutral['V_diff'] = data_neutral['V'].sub(3)
data_neutral['A_diff'] = data_neutral['A'].sub(3)
data_neutral['D_diff'] = data_neutral['D'].sub(3)

#JOY
data_joy['V_diff'] = data_joy['V'].sub(3.992)
data_joy['A_diff'] = data_joy['A'].sub(3.516)
data_joy['D_diff'] = data_joy['D'].sub(3.295)

#SURPRISE
data_surprise['V_diff'] = data_surprise['V'].sub(3.380)
data_surprise['A_diff'] = data_surprise['A'].sub(3.839)
data_surprise['D_diff'] = data_surprise['D'].sub(2.479)

#SADNESS
data_sadness['V_diff'] = data_sadness['V'].sub(1.629)
data_sadness['A_diff'] = data_sadness['A'].sub(3.159)
data_sadness['D_diff'] = data_sadness['D'].sub(2.139)

#ANGER
data_anger['V_diff'] = data_anger['V'].sub(1.629)
data_anger['A_diff'] = data_anger['A'].sub(3.159)
data_anger['D_diff'] = data_anger['D'].sub(2.139)

#FEAR
data_fear['V_diff'] = data_fear['V'].sub(1.612)
data_fear['A_diff'] = data_fear['A'].sub(3.720)
data_fear['D_diff'] = data_fear['D'].sub(1.969)

#DISGUST
data_disgust['V_diff'] = data_disgust['V'].sub(3.380)
data_disgust['A_diff'] = data_disgust['A'].sub(3.839)
data_disgust['D_diff'] = data_disgust['D'].sub(2.479)


columns = ['V_diff', 'A_diff', 'D_diff']
fig, axs = plt.subplots(3, 1, figsize=(6, 12))
for i, column in enumerate(columns):
    axs[i].plot(data_neutral[column], data_neutral['predicted_emotion'], 'o')
    axs[i].plot(data_joy[column], data_joy['predicted_emotion'], 'o')
    axs[i].plot(data_surprise[column], data_surprise['predicted_emotion'], 'o')
    axs[i].plot(data_sadness[column], data_sadness['predicted_emotion'], 'o')
    axs[i].plot(data_anger[column], data_anger['predicted_emotion'], 'o')
    axs[i].plot(data_fear[column], data_fear['predicted_emotion'], 'o')
    axs[i].plot(data_disgust[column], data_disgust['predicted_emotion'], 'o')
    axs[i].set_title(f'{column} Values Distance', fontsize=12, fontweight='bold', color='black')
    axs[i].set_xlabel(column)
    axs[i].set_ylabel('Emotion')
    axs[i].set_xlim(-3, 3)
plt.tight_layout()
plt.show()




#NEUTRAL
#Text characteristics
primary_neutral['V_diff'] = primary_neutral['V'].sub(3).abs()
primary_neutral['A_diff'] = primary_neutral['A'].sub(3).abs()
primary_neutral['D_diff'] = primary_neutral['D'].sub(3).abs()
close_predictions = primary_neutral.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_neutral.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_neutral[close_predictions]
distant_predictions = primary_neutral[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Neutral')
plt.show()

print("Neutral Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nNeutral Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nNeutral Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {distant_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nNeutral Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Neutral - Text characteristics")
plt.legend()
plt.show()

#Text complexity indices
print("Neutral Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nNeutral Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nNeutral Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nNeutral Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Neutral - Text complexity indices")
plt.legend()
plt.show()



#JOY
primary_joy['V_diff'] = primary_joy['V'].sub(3).abs()
primary_joy['A_diff'] = primary_joy['A'].sub(3).abs()
primary_joy['D_diff'] = primary_joy['D'].sub(3).abs()
close_predictions = primary_joy.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_joy.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_joy[close_predictions]
distant_predictions = primary_joy[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Joy')
plt.show()

print("Joy Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nJoy Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nJoy Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {distant_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nJoy Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Joy")
plt.legend()
plt.show()

#Text complexity indices
print("Joy Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nJoy Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nJoy Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nJoy Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Joy - Text complexity indices")
plt.legend()
plt.show()



#SURPRISE
primary_surprise['V_diff'] = primary_surprise['V'].sub(3).abs()
primary_surprise['A_diff'] = primary_surprise['A'].sub(3).abs()
primary_surprise['D_diff'] = primary_surprise['D'].sub(3).abs()
close_predictions = primary_surprise.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_surprise.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_surprise[close_predictions]
distant_predictions = primary_surprise[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Surprise')
plt.show()

print("Surprise Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nSurprise Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nSurprise Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {distant_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nSurprise Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Surprise")
plt.legend()
plt.show()

#Text complexity indices
print("Surprise Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nSurprise Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nSurprise Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nSurprise Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Surprise - Text complexity indices")
plt.legend()
plt.show()



#SADNESS
primary_sadness['V_diff'] = primary_sadness['V'].sub(3).abs()
primary_sadness['A_diff'] = primary_sadness['A'].sub(3).abs()
primary_sadness['D_diff'] = primary_sadness['D'].sub(3).abs()
close_predictions = primary_sadness.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_sadness.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_sadness[close_predictions]
distant_predictions = primary_sadness[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Sadness')
plt.show()

print("Sadness Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nSadness Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nSadness Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {distant_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nSadness Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Sadness")
plt.legend()
plt.show()

#Text complexity indices
print("Sadness Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nSadness Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nSadness Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nSadness Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Sadness - Text complexity indices")
plt.legend()
plt.show()




#ANGER
primary_anger['V_diff'] = primary_anger['V'].sub(3).abs()
primary_anger['A_diff'] = primary_anger['A'].sub(3).abs()
primary_anger['D_diff'] = primary_anger['D'].sub(3).abs()
close_predictions = primary_anger.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_anger.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_anger[close_predictions]
distant_predictions = primary_anger[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Anger')
plt.show()

print("Anger Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nAnger Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nAnger Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {distant_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nAnger Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Anger")
plt.legend()
plt.show()

#Text complexity indices
print("Anger Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nAnger Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nAnger Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nAnger Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Anger - Text complexity indices")
plt.legend()
plt.show()




#FEAR
primary_fear['V_diff'] = primary_fear['V'].sub(3).abs()
primary_fear['A_diff'] = primary_fear['A'].sub(3).abs()
primary_fear['D_diff'] = primary_fear['D'].sub(3).abs()
close_predictions = primary_fear.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_fear.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_fear[close_predictions]
distant_predictions = primary_fear[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Fear')
plt.show()

print("Fear Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nFear Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nFear Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nFear Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Fear")
plt.legend()
plt.show()

#Text complexity indices
print("Fear Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nFear Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nFear Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nFear Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Fear - Text complexity indices")
plt.legend()
plt.show()




#DISGUST
primary_disgust['V_diff'] = primary_disgust['V'].sub(3).abs()
primary_disgust['A_diff'] = primary_disgust['A'].sub(3).abs()
primary_disgust['D_diff'] = primary_disgust['D'].sub(3).abs()
close_predictions = primary_disgust.apply(lambda row:
                                   (row['V_diff'] <= 0.75) &
                                   (row['A_diff'] <= 0.75) &
                                   (row['D_diff'] <= 0.75), axis=1)
distant_predictions = primary_disgust.apply(lambda row:
                                   (row['V_diff'] > 0.75) |
                                   (row['A_diff'] > 0.75) |
                                   (row['D_diff'] > 0.75), axis=1)
close_predictions = primary_disgust[close_predictions]
distant_predictions = primary_disgust[distant_predictions]

print(close_predictions)
print(distant_predictions)

#close and distant prediction plots
dominanceC = close_predictions['D']
valenceC = close_predictions['V']
arousalC = close_predictions['A']
dominanceD = distant_predictions['D']
valenceD = distant_predictions['V']
arousalD = distant_predictions['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousalC, valenceC, dominanceC, color='purple', alpha=0.2)
ax.scatter(arousalD, valenceD, dominanceD, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Disgust')
plt.show()

print("Disgust Emotion - Close Predictions Analysis:")
print(f" - Average syllable count: {close_predictions['syllable_count'].mean()}")
print(f" - Average word count: {close_predictions['word_count'].mean()}")
print(f" - Average letter count: {close_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {close_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {close_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {close_predictions['std_dev_word_length'].mean()}")
print("\nDisgust Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nDisgust Emotion - Distant Predictions Analysis:")
print(f" - Average syllable count: {distant_predictions['syllable_count'].mean()}")
print(f" - Average word count: {distant_predictions['word_count'].mean()}")
print(f" - Average letter count: {distant_predictions['letter_count'].mean()}")
print(f" - Average Verb Count: {distant_predictions['unique_words'].mean()}")
print(f" - Average mean word length: {distant_predictions['mean_word_length'].mean()}")
print(f" - Average std dev word length: {distant_predictions['std_dev_word_length'].mean()}")
print("\nDisgust Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['syllable_count'].mean(), close_predictions['word_count'].mean(), close_predictions['letter_count'].mean(), close_predictions['unique_words'].mean(), close_predictions['mean_word_length'].mean(), close_predictions['std_dev_word_length'].mean()]
distant_pred = [distant_predictions['syllable_count'].mean(), distant_predictions['word_count'].mean(), distant_predictions['letter_count'].mean(), distant_predictions['unique_words'].mean(), distant_predictions['mean_word_length'].mean(), distant_predictions['std_dev_word_length'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['syllable_count', 'word_count', 'letter_count', 'unique_words', 'mean_word_length', 'std_dev_word_length'])
plt.title("Disgust")
plt.legend()
plt.show()

#Text complexity indices
print("Disgust Emotion - Close Predictions Analysis:")
print(f" - Average flesch_reading_ease: {close_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {close_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {close_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {close_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {close_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {close_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {close_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {close_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {close_predictions['reading_time'].mean()}")
print("\nDisgust Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nDisgust Emotion - Distant Predictions Analysis:")
print(f" - Average flesch_reading_ease: {distant_predictions['flesch_reading_ease'].mean()}")
print(f" - Average flesch_kincaid_grade: {distant_predictions['flesch_kincaid_grade'].mean()}")
print(f" - Average gunning_fog: {distant_predictions['gunning_fog'].mean()}")
print(f" - Average automated_readability_index: {distant_predictions['automated_readability_index'].mean()}")
print(f" - Average coleman_liau_index: {distant_predictions['coleman_liau_index'].mean()}")
print(f" - Average linsear_write_formula: {distant_predictions['linsear_write_formula'].mean()}")
print(f" - Average dale_chall_readability_score: {distant_predictions['dale_chall_readability_score'].mean()}")
print(f" - Average mcalpine_eflaw: {distant_predictions['mcalpine_eflaw'].mean()}")
print(f" - Average reading_time: {distant_predictions['reading_time'].mean()}")
print("\nDisgust Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(14, 14))

close_pred = [close_predictions['flesch_reading_ease'].mean(), close_predictions['flesch_kincaid_grade'].mean(), close_predictions['gunning_fog'].mean(), close_predictions['automated_readability_index'].mean(), close_predictions['coleman_liau_index'].mean(), close_predictions['linsear_write_formula'].mean(), close_predictions['dale_chall_readability_score'].mean(), close_predictions['mcalpine_eflaw'].mean(), close_predictions['reading_time'].mean()]
distant_pred = [distant_predictions['flesch_reading_ease'].mean(), distant_predictions['flesch_kincaid_grade'].mean(), distant_predictions['gunning_fog'].mean(), distant_predictions['automated_readability_index'].mean(), distant_predictions['coleman_liau_index'].mean(), distant_predictions['linsear_write_formula'].mean(), distant_predictions['dale_chall_readability_score'].mean(), distant_predictions['mcalpine_eflaw'].mean(), distant_predictions['reading_time'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'automated_readability_idx', 'coleman_liau_idx', 'linsear_write_formula', 'dale_chall_readability', 'mcalpine_eflaw', 'reading_time'], rotation=45, ha='right')
plt.title("Disgust - Text complexity indices")
plt.legend()
plt.show()

