import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('ekman_predictions_arpanghoshal_EkmanClassifier_characteristics_analysis.csv')
data_neutral = data.loc[data['predicted_emotion'] == 'neutral']
data_joy = data.loc[data['predicted_emotion'] == 'joy']
data_surprise = data.loc[data['predicted_emotion'] == 'surprise']
data_sadness = data.loc[data['predicted_emotion'] == 'sadness']
data_anger = data.loc[data['predicted_emotion'] == 'anger']
data_fear = data.loc[data['predicted_emotion'] == 'fear']
data_disgust = data.loc[data['predicted_emotion'] == 'disgust']



#NEUTRAL
data_neutral['V_diff_neutral'] = data_neutral['V'].sub(3).abs()
data_neutral['A_diff_neutral'] = data_neutral['A'].sub(3).abs()
data_neutral['D_diff_neutral'] = data_neutral['D'].sub(3).abs()
close_predictions = data_neutral.apply(lambda row:
                                   (row['V_diff_neutral'] <= 0.75) &
                                   (row['A_diff_neutral'] <= 0.75) &
                                   (row['D_diff_neutral'] <= 0.75), axis=1)
distant_predictions = data_neutral.apply(lambda row:
                                   (row['V_diff_neutral'] > 0.75) |
                                   (row['A_diff_neutral'] > 0.75) |
                                   (row['D_diff_neutral'] > 0.75), axis=1)
close_predictions = data_neutral[close_predictions]
distant_predictions = data_neutral[distant_predictions]

print(close_predictions)
print(distant_predictions)

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
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nNeutral Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nNeutral Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nNeutral Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Neutral")
plt.legend()
plt.show()




#JOY
data_joy['V_diff_joy'] = data_joy['V'].sub(3.992).abs()
data_joy['A_diff_joy'] = data_joy['A'].sub(3.516).abs()
data_joy['D_diff_joy'] = data_joy['D'].sub(3.295).abs()
close_predictions = data_joy.apply(lambda row:
                                   (row['V_diff_joy'] <= 0.75) &
                                   (row['A_diff_joy'] <= 0.75) &
                                   (row['D_diff_joy'] <= 0.75), axis=1)
distant_predictions = data_joy.apply(lambda row:
                                   (row['V_diff_joy'] > 0.75) |
                                   (row['A_diff_joy'] > 0.75) |
                                   (row['D_diff_joy'] > 0.75), axis=1)
close_predictions = data_joy[close_predictions]
distant_predictions = data_joy[distant_predictions]

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
ax.scatter([3.516], [3.992], [3.295], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Joy')
plt.show()

print("Joy Emotion - Close Predictions Analysis:")
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nJoy Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nJoy Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nJoy Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Joy")
plt.legend()
plt.show()



#SURPRISE
data_surprise['V_diff_surprise'] = data_surprise['V'].sub(3.380).abs()
data_surprise['A_diff_surprise'] = data_surprise['A'].sub(3.839).abs()
data_surprise['D_diff_surprise'] = data_surprise['D'].sub(2.479).abs()
close_predictions = data_surprise.apply(lambda row:
                                   (row['V_diff_surprise'] <= 0.75) &
                                   (row['A_diff_surprise'] <= 0.75) &
                                   (row['D_diff_surprise'] <= 0.75), axis=1)
distant_predictions = data_surprise.apply(lambda row:
                                   (row['V_diff_surprise'] > 0.75) |
                                   (row['A_diff_surprise'] > 0.75) |
                                   (row['D_diff_surprise'] > 0.75), axis=1)
close_predictions = data_surprise[close_predictions]
distant_predictions = data_surprise[distant_predictions]

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
ax.scatter([3.839], [3.380], [2.479], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Surprise')
plt.show()

print("Surprise Emotion - Close Predictions Analysis:")
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nSurprise Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nSurprise Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nSurprise Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Surprise")
plt.legend()
plt.show()



#SADNESS
data_sadness['V_diff_sadness'] = data_sadness['V'].sub(1.629).abs()
data_sadness['A_diff_sadness'] = data_sadness['A'].sub(3.159).abs()
data_sadness['D_diff_sadness'] = data_sadness['D'].sub(2.139).abs()
close_predictions = data_sadness.apply(lambda row:
                                   (row['V_diff_sadness'] <= 0.75) &
                                   (row['A_diff_sadness'] <= 0.75) &
                                   (row['D_diff_sadness'] <= 0.75), axis=1)
distant_predictions = data_sadness.apply(lambda row:
                                   (row['V_diff_sadness'] > 0.75) |
                                   (row['A_diff_sadness'] > 0.75) |
                                   (row['D_diff_sadness'] > 0.75), axis=1)
close_predictions = data_sadness[close_predictions]
distant_predictions = data_sadness[distant_predictions]
print(close_predictions)
print(distant_predictions)
print(data_sadness[['V', 'A', 'D', 'V_diff_sadness', 'A_diff_sadness', 'D_diff_sadness']].head())

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
ax.scatter([3.159], [1.629], [2.139], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Sadness')
plt.show()

print("Sadness Emotion - Close Predictions Analysis:")
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nSadness Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nSadness Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nSadness Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Sadness")
plt.legend()
plt.show()



#ANGER
data_anger['V_diff_anger'] = data_anger['V'].sub(1.969).abs()
data_anger['A_diff_anger'] = data_anger['A'].sub(3.839).abs()
data_anger['D_diff_anger'] = data_anger['D'].sub(3.278).abs()
close_predictions = data_anger.apply(lambda row:
                                   (row['V_diff_anger'] <= 0.75) &
                                   (row['A_diff_anger'] <= 0.75) &
                                   (row['D_diff_anger'] <= 0.75), axis=1)
distant_predictions = data_anger.apply(lambda row:
                                   (row['V_diff_anger'] > 0.75) |
                                   (row['A_diff_anger'] > 0.75) |
                                   (row['D_diff_anger'] > 0.75), axis=1)
close_predictions = data_anger[close_predictions]
distant_predictions = data_anger[distant_predictions]

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
ax.scatter([3.839], [1.969], [3.278], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Anger')
plt.show()

print("Anger Emotion - Close Predictions Analysis:")
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nAnger Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nAnger Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nAnger Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Anger")
plt.legend()
plt.show()



#FEAR
data_fear['V_diff_fear'] = data_fear['V'].sub(1.612).abs()
data_fear['A_diff_fear'] = data_fear['A'].sub(3.720).abs()
data_fear['D_diff_fear'] = data_fear['D'].sub(1.969).abs()
close_predictions = data_fear.apply(lambda row:
                                   (row['V_diff_fear'] <= 0.75) &
                                   (row['A_diff_fear'] <= 0.75) &
                                   (row['D_diff_fear'] <= 0.75), axis=1)
distant_predictions = data_fear.apply(lambda row:
                                   (row['V_diff_fear'] > 0.75) |
                                   (row['A_diff_fear'] > 0.75) |
                                   (row['D_diff_fear'] > 0.75), axis=1)
close_predictions = data_fear[close_predictions]
distant_predictions = data_fear[distant_predictions]

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
ax.scatter([3.720], [1.612], [1.969], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Fear')
plt.show()

print("Fear Emotion - Close Predictions Analysis:")
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nFear Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nFear Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nFear Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Fear")
plt.legend()
plt.show()



#DISGUST
data_disgust['V_diff_disgust'] = data_disgust['V'].sub(3).abs()
data_disgust['A_diff_disgust'] = data_disgust['A'].sub(3).abs()
data_disgust['D_diff_disgust'] = data_disgust['D'].sub(3).abs()
close_predictions = data_disgust.apply(lambda row:
                                   (row['V_diff_disgust'] <= 0.75) &
                                   (row['A_diff_disgust'] <= 0.75) &
                                   (row['D_diff_disgust'] <= 0.75), axis=1)
distant_predictions = data_disgust.apply(lambda row:
                                   (row['V_diff_disgust'] > 0.75) |
                                   (row['A_diff_disgust'] > 0.75) |
                                   (row['D_diff_disgust'] > 0.75), axis=1)
close_predictions = data_disgust[close_predictions]
distant_predictions = data_disgust[distant_predictions]

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
print(f" - Average Syllable Count: {close_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {close_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {close_predictions['number_of_verbs'].mean()}")
print("\nDisgust Emotion - Close Predictions Statistics:")
print(close_predictions.describe())

print("\nDisgust Emotion - Distant Predictions Analysis:")
print(f" - Average Syllable Count: {distant_predictions['number_of_syllables'].mean()}")
print(f" - Average Phrase Length: {distant_predictions['phrase_length'].mean()}")
print(f" - Average Verb Count: {distant_predictions['number_of_verbs'].mean()}")
print("\nDisgust Emotion - Distant Predictions Statistics:")
print(distant_predictions.describe())

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

close_pred = [close_predictions['number_of_syllables'].mean(), close_predictions['phrase_length'].mean(), close_predictions['number_of_verbs'].mean()]
distant_pred = [distant_predictions['number_of_syllables'].mean(), distant_predictions['phrase_length'].mean(), distant_predictions['number_of_verbs'].mean()]

br1 = np.arange(len(close_pred))
br2 = [x + barWidth for x in br1]

plt.bar(br1, close_pred, color ='purple', width = barWidth,
        edgecolor ='grey', label ='Close')
plt.bar(br2, distant_pred, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Distant')

plt.xticks([r + barWidth for r in range(len(close_pred))],
           ['Numero di sillabe', 'Lunghezza frase', 'Numero di verbi'])
plt.title("Disgust")
plt.legend()
plt.show()






'''
#NEUTRAL
data_neutral['V_diff_neutral'] = data_neutral['V'].sub(3).abs()
data_neutral['A_diff_neutral'] = data_neutral['A'].sub(3).abs()
data_neutral['D_diff_neutral'] = data_neutral['D'].sub(3).abs()
filtered_data = data_neutral.apply(lambda row:
                                   (row['V_diff_neutral'] < 0.75) &
                                   (row['A_diff_neutral'] < 0.75) &
                                   (row['D_diff_neutral'] < 0.75), axis=1)
result = data_neutral[filtered_data]
print(result)
print(data_neutral[['V', 'A', 'D', 'V_diff_neutral', 'A_diff_neutral', 'D_diff_neutral']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Neutral')
plt.show()



#JOY
data_joy['V_diff_joy'] = data_joy['V'].sub(3.992).abs()
data_joy['A_diff_joy'] = data_joy['A'].sub(3.516).abs()
data_joy['D_diff_joy'] = data_joy['D'].sub(3.295).abs()
filtered_data = data_joy.apply(lambda row:
                              (row['V_diff_joy'] < 0.75) &
                              (row['A_diff_joy'] < 0.75) &
                              (row['D_diff_joy'] < 0.75), axis=1)
result = data_joy[filtered_data]
print(result)
print(data_joy[['V', 'A', 'D', 'V_diff_joy', 'A_diff_joy', 'D_diff_joy']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.516], [3.992], [3.295], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Joy')
plt.show()



#SURPRISE
data_surprise['V_diff_surprise'] = data_surprise['V'].sub(3.380).abs()
data_surprise['A_diff_surprise'] = data_surprise['A'].sub(3.839).abs()
data_surprise['D_diff_surprise'] = data_surprise['D'].sub(2.479).abs()
filtered_data = data_surprise.apply(lambda row:
                                    (row['V_diff_surprise'] < 0.75) &
                                    (row['A_diff_surprise'] < 0.75) &
                                    (row['D_diff_surprise'] < 0.75), axis=1)
result = data_surprise[filtered_data]
print(result)
print(data_surprise[['V', 'A', 'D', 'V_diff_surprise', 'A_diff_surprise', 'D_diff_surprise']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.839], [3.380], [2.479], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Surprise')
plt.show()



#SADNESS
data_sadness['V_diff_sadness'] = data_sadness['V'].sub(1.629).abs()
data_sadness['A_diff_sadness'] = data_sadness['A'].sub(3.159).abs()
data_sadness['D_diff_sadness'] = data_sadness['D'].sub(2.139).abs()
filtered_data = data_sadness.apply(lambda row:
                                  (row['V_diff_sadness'] < 0.75) &
                                  (row['A_diff_sadness'] < 0.75) &
                                  (row['D_diff_sadness'] < 0.75), axis=1)
result = data_sadness[filtered_data]
print(result)
print(data_sadness[['V', 'A', 'D', 'V_diff_sadness', 'A_diff_sadness', 'D_diff_sadness']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.159], [1.629], [2.139], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Sadness')
plt.show()



#ANGER
data_anger['V_diff_anger'] = data_anger['V'].sub(1.969).abs()
data_anger['A_diff_anger'] = data_anger['A'].sub(3.839).abs()
data_anger['D_diff_anger'] = data_anger['D'].sub(3.278).abs()
filtered_data = data_anger.apply(lambda row:
                                (row['V_diff_anger'] < 0.75) &
                                (row['A_diff_anger'] < 0.75) &
                                (row['D_diff_anger'] < 0.75), axis=1)
result = data_anger[filtered_data]
print(result)
print(data_anger[['V', 'A', 'D', 'V_diff_anger', 'A_diff_anger', 'D_diff_anger']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.839], [1.969], [3.278], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Anger')
plt.show()



#FEAR
data_fear['V_diff_fear'] = data_fear['V'].sub(1.612).abs()
data_fear['A_diff_fear'] = data_fear['A'].sub(3.720).abs()
data_fear['D_diff_fear'] = data_fear['D'].sub(1.969).abs()
filtered_data = data_fear.apply(lambda row:
                               (row['V_diff_fear'] < 0.75) &
                               (row['A_diff_fear'] < 0.75) &
                               (row['D_diff_fear'] < 0.75), axis=1)
result = data_fear[filtered_data]
print(result)
print(data_fear[['V', 'A', 'D', 'V_diff_fear', 'A_diff_fear', 'D_diff_fear']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.720], [1.612], [1.969], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Fear')
plt.show()



#DISGUST
data_disgust['V_diff_disgust'] = data_disgust['V'].sub(3).abs()
data_disgust['A_diff_disgust'] = data_disgust['A'].sub(3).abs()
data_disgust['D_diff_disgust'] = data_disgust['D'].sub(3).abs()
filtered_data = data_disgust.apply(lambda row:
                                 (row['V_diff_disgust'] < 0.75) &
                                 (row['A_diff_disgust'] < 0.75) &
                                 (row['D_diff_disgust'] < 0.75), axis=1)
result = data_disgust[filtered_data]
print(result)
print(data_disgust[['V', 'A', 'D', 'V_diff_disgust', 'A_diff_disgust', 'D_diff_disgust']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Disgust')
plt.show()







#NEUTRAL
data_neutral['V_diff_neutral'] = data_neutral['V'].sub(3).abs()
data_neutral['A_diff_neutral'] = data_neutral['A'].sub(3).abs()
data_neutral['D_diff_neutral'] = data_neutral['D'].sub(3).abs()
filtered_data = data_neutral.apply(lambda row:
                                   (row['V_diff_neutral'] < 1) &
                                   (row['A_diff_neutral'] < 1) &
                                   (row['D_diff_neutral'] < 1), axis=1)
result = data_neutral[filtered_data]
print(result)
print(data_neutral[['V', 'A', 'D', 'V_diff_neutral', 'A_diff_neutral', 'D_diff_neutral']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Neutral')
plt.show()



#JOY
data_joy['V_diff_joy'] = data_joy['V'].sub(3.992).abs()
data_joy['A_diff_joy'] = data_joy['A'].sub(3.516).abs()
data_joy['D_diff_joy'] = data_joy['D'].sub(3.295).abs()
filtered_data = data_joy.apply(lambda row:
                              (row['V_diff_joy'] < 1) &
                              (row['A_diff_joy'] < 1) &
                              (row['D_diff_joy'] < 1), axis=1)
result = data_joy[filtered_data]
print(result)
print(data_joy[['V', 'A', 'D', 'V_diff_joy', 'A_diff_joy', 'D_diff_joy']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.516], [3.992], [3.295], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Joy')
plt.show()



#SURPRISE
data_surprise['V_diff_surprise'] = data_surprise['V'].sub(3.380).abs()
data_surprise['A_diff_surprise'] = data_surprise['A'].sub(3.839).abs()
data_surprise['D_diff_surprise'] = data_surprise['D'].sub(2.479).abs()
filtered_data = data_surprise.apply(lambda row:
                                    (row['V_diff_surprise'] < 1) &
                                    (row['A_diff_surprise'] < 1) &
                                    (row['D_diff_surprise'] < 1), axis=1)
result = data_surprise[filtered_data]
print(result)
print(data_surprise[['V', 'A', 'D', 'V_diff_surprise', 'A_diff_surprise', 'D_diff_surprise']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.839], [3.380], [2.479], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Surprise')
plt.show()



#SADNESS
data_sadness['V_diff_sadness'] = data_sadness['V'].sub(1.629).abs()
data_sadness['A_diff_sadness'] = data_sadness['A'].sub(3.159).abs()
data_sadness['D_diff_sadness'] = data_sadness['D'].sub(2.139).abs()
filtered_data = data_sadness.apply(lambda row:
                                  (row['V_diff_sadness'] < 1) &
                                  (row['A_diff_sadness'] < 1) &
                                  (row['D_diff_sadness'] < 1), axis=1)
result = data_sadness[filtered_data]
print(result)
print(data_sadness[['V', 'A', 'D', 'V_diff_sadness', 'A_diff_sadness', 'D_diff_sadness']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.159], [1.629], [2.139], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Sadness')
plt.show()



#ANGER
data_anger['V_diff_anger'] = data_anger['V'].sub(1.969).abs()
data_anger['A_diff_anger'] = data_anger['A'].sub(3.839).abs()
data_anger['D_diff_anger'] = data_anger['D'].sub(3.278).abs()
filtered_data = data_anger.apply(lambda row:
                                (row['V_diff_anger'] < 1) &
                                (row['A_diff_anger'] < 1) &
                                (row['D_diff_anger'] < 1), axis=1)
result = data_anger[filtered_data]
print(result)
print(data_anger[['V', 'A', 'D', 'V_diff_anger', 'A_diff_anger', 'D_diff_anger']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.839], [1.969], [3.278], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Anger')
plt.show()



#FEAR
data_fear['V_diff_fear'] = data_fear['V'].sub(1.612).abs()
data_fear['A_diff_fear'] = data_fear['A'].sub(3.720).abs()
data_fear['D_diff_fear'] = data_fear['D'].sub(1.969).abs()
filtered_data = data_fear.apply(lambda row:
                               (row['V_diff_fear'] < 1) &
                               (row['A_diff_fear'] < 1) &
                               (row['D_diff_fear'] < 1), axis=1)
result = data_fear[filtered_data]
print(result)
print(data_fear[['V', 'A', 'D', 'V_diff_fear', 'A_diff_fear', 'D_diff_fear']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.720], [1.612], [1.969], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Fear')
plt.show()



#DISGUST
data_disgust['V_diff_disgust'] = data_disgust['V'].sub(3).abs()
data_disgust['A_diff_disgust'] = data_disgust['A'].sub(3).abs()
data_disgust['D_diff_disgust'] = data_disgust['D'].sub(3).abs()
filtered_data = data_disgust.apply(lambda row:
                                 (row['V_diff_disgust'] < 1) &
                                 (row['A_diff_disgust'] < 1) &
                                 (row['D_diff_disgust'] < 1), axis=1)
result = data_disgust[filtered_data]
print(result)
print(data_disgust[['V', 'A', 'D', 'V_diff_disgust', 'A_diff_disgust', 'D_diff_disgust']].head())

dominance = result['D']
valence = result['V']
arousal = result['A']
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(1, 5)
ax.set_ylim(1, 5)
ax.set_zlim(1, 5)
ax.scatter(arousal, valence, dominance, color='blue', alpha=0.2)
ax.scatter([3.000], [3.000], [3.000], color='red', s=40, marker='o')
ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Dominance')
plt.title('Disgust')
plt.show()







riferimenti_joy = np.array([3.992, 3.516, 3.295])
data_joy['VAD_diff_joy'] = data_joy[['V', 'A', 'D']].sub(riferimenti_joy).abs().sum(axis=1)
print(data_joy[['text', 'predicted_emotion', 'V', 'A', 'D', 'VAD_diff_joy']].head())
diff_stats = data_joy['VAD_diff_joy'].describe()
print(diff_stats)

plt.figure(figsize=(10, 6))
data_joy['VAD_diff_joy'].hist(bins=20, edgecolor='black')
plt.title('Distribuzione delle differenze VAD rispetto ai valori di riferimento')
plt.xlabel('Somma delle differenze assolute')
plt.ylabel('Frequenza')
plt.show()



riferimenti_surprise = np.array([3.380, 3.839, 2.479])
data_surprise['VAD_diff_surprise'] = data_surprise[['V', 'A', 'D']].sub(riferimenti_surprise).abs().sum(axis=1)
print(data_surprise[['text', 'predicted_emotion', 'V', 'A', 'D', 'VAD_diff_surprise']].head())
diff_stats = data_surprise['VAD_diff_surprise'].describe()
print(diff_stats)

plt.figure(figsize=(10, 6))
data_surprise['VAD_diff_surprise'].hist(bins=20, edgecolor='black')
plt.title('Distribuzione delle differenze VAD rispetto ai valori di riferimento')
plt.xlabel('Somma delle differenze assolute')
plt.ylabel('Frequenza')
plt.show()



riferimenti_sadness = np.array([1.629, 3.159, 2.139])
data_sadness['VAD_diff_sadness'] = data_sadness[['V', 'A', 'D']].sub(riferimenti_sadness).abs().sum(axis=1)
print(data_sadness[['text', 'predicted_emotion', 'V', 'A', 'D', 'VAD_diff_sadness']].head())
diff_stats = data_sadness['VAD_diff_sadness'].describe()
print(diff_stats)

plt.figure(figsize=(10, 6))
data_sadness['VAD_diff_sadness'].hist(bins=20, edgecolor='black')
plt.title('Distribuzione delle differenze VAD rispetto ai valori di riferimento')
plt.xlabel('Somma delle differenze assolute')
plt.ylabel('Frequenza')
plt.show()



riferimenti_anger = np.array([1.969, 3.839, 3.278])
data_anger['VAD_diff_anger'] = data_anger[['V', 'A', 'D']].sub(riferimenti_anger).abs().sum(axis=1)
print(data_anger[['text', 'predicted_emotion', 'V', 'A', 'D', 'VAD_diff_anger']].head())
diff_stats = data_anger['VAD_diff_anger'].describe()
print(diff_stats)

plt.figure(figsize=(10, 6))
data_anger['VAD_diff_anger'].hist(bins=20, edgecolor='black')
plt.title('Distribuzione delle differenze VAD rispetto ai valori di riferimento')
plt.xlabel('Somma delle differenze assolute')
plt.ylabel('Frequenza')
plt.show()



riferimenti_fear = np.array([1.612, 3.720, 1.969])
data_fear['VAD_diff_fear'] = data_fear[['V', 'A', 'D']].sub(riferimenti_fear).abs().sum(axis=1)
print(data_fear[['text', 'predicted_emotion', 'V', 'A', 'D', 'VAD_diff_fear']].head())
diff_stats = data_fear['VAD_diff_fear'].describe()
print(diff_stats)

plt.figure(figsize=(10, 6))
data_fear['VAD_diff_fear'].hist(bins=20, edgecolor='black')
plt.title('Distribuzione delle differenze VAD rispetto ai valori di riferimento')
plt.xlabel('Somma delle differenze assolute')
plt.ylabel('Frequenza')
plt.show()



riferimenti_disgust = np.array([1.680, 3.295, 2.887])
data_disgust['VAD_diff_disgust'] = data_disgust[['V', 'A', 'D']].sub(riferimenti_disgust).abs().sum(axis=1)
print(data_disgust[['text', 'predicted_emotion', 'V', 'A', 'D', 'VAD_diff_disgust']].head())
diff_stats = data_disgust['VAD_diff_disgust'].describe()
print(diff_stats)

plt.figure(figsize=(10, 6))
data_disgust['VAD_diff_disgust'].hist(bins=20, edgecolor='black')
plt.title('Distribuzione delle differenze VAD rispetto ai valori di riferimento')
plt.xlabel('Somma delle differenze assolute')
plt.ylabel('Frequenza')
plt.show()
'''


