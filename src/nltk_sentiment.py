# 调库直接进行极性分类
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn import metrics

# Read the CSV file
# df_val = pd.read_csv('data/val_washed_data.csv', header=None, encoding='ISO-8859-1')
df_val = pd.read_csv('data/testdata.csv', header=None, encoding='ISO-8859-1')

# Function to classify polarity of a sentence
def classify_polarity(sentence):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(sentence)["compound"]
    
    if sentiment_score >= 0:
        return 1  # Positive
    elif sentiment_score < 0:
        return 0  # Negative

# Apply the classification to the DataFrame
df_val['Polarity'] = df_val[5].apply(classify_polarity)

X_val = df_val.iloc[:, 5].copy()
y_val = df_val.iloc[:, 0].copy()
y_val.replace(4,1,inplace=True)

false_neg = 0
false_pos = 0
for i in range(len(X_val)):
    if df_val['Polarity'][i] > 0:
        if y_val[i] != 1:
            print("error index: {} it should be 0 sentence: {}".format(i, X_val[i]))
    elif df_val['Polarity'][i] == 0:
        if y_val[i] != 0:
            print("error index: {} it should be 1 sentence: {}".format(i, X_val[i]))

accuracy = metrics.accuracy_score(y_val, df_val['Polarity'])
precision = metrics.precision_score(y_val, df_val['Polarity'], average='binary')
recall = metrics.recall_score(y_val, df_val['Polarity'], average='binary')
f1 = metrics.f1_score(y_val, df_val['Polarity'], average='binary')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# Display the DataFrame with the added 'Polarity' column
# print(df_val)
