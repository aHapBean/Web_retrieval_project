import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import re
import os
import nltk

model_filename = './model/naive_bayes/nb_moble.joblib'
vectorizer_filename = './model/naive_bayes/bayes_vectorizer.joblib'

# file_path = 'training.1600000.processed.noemoticon.csv'
LIMIT = int(16e5)
# file_path = 'washed_data.csv'
file_path = 'training.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.iloc[:LIMIT, 5].copy()
y = df.iloc[:LIMIT, 0].copy()
print("start training...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer(tokenizer=word_tokenize)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Multinomial Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test_vec)

# Print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

if not os.path.exists('./model/naive_bayes'):
    os.makedirs('./model/naive_bayes')
joblib.dump(nb_model, model_filename) # NOTE here
joblib.dump(vectorizer, vectorizer_filename)

# NOTE 测试部分
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)
# test_df = pd.read_csv('data/val_washed_data.csv', header=None, encoding='ISO-8859-1')
test_df = pd.read_csv('test.csv', header=None, encoding='ISO-8859-1')

test_data = test_df[5]  # 选择处理后的文本列
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)
accuracy = accuracy_score(test_df[0], test_pred)  # 使用原始标签列


print(accuracy)

# 100000 n_es 8:2
# val set acc 75.785%
# test set acc 76.602%