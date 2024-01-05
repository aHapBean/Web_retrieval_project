import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import re
import os
import nltk

model_filename = '../model/naive_bayes/nb_moble.joblib'
vectorizer_filename = '../model/naive_bayes/bayes_vectorizer.joblib'

# file_path = 'training.1600000.processed.noemoticon.csv'
LIMIT = int(100000)
file_path = '../data/washed_train_data.csv'
df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.iloc[:LIMIT, 5].copy()
y = df.iloc[:LIMIT, 0].copy()
print("start training...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
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

# Calculate precision, recall, and f1
precision = precision_score(y_test, y_pred, average='binary')  # binary classification
recall = recall_score(y_test, y_pred, average='binary')  # binary classification
f1 = f1_score(y_test, y_pred, average='binary')  # binary classification

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

if not os.path.exists('../model/naive_bayes'):
    os.makedirs('../model/naive_bayes')
joblib.dump(nb_model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

# NOTE 测试部分
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)
test_df = pd.read_csv('../data/washed_test_data.csv', header=None, encoding='ISO-8859-1')    # raw 表示简单清洗

test_data = test_df[5]  # 选择处理后的文本列
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)

# Calculate accuracy for the test set
accuracy_test = accuracy_score(test_df[0], test_pred)
print("Test Accuracy:", accuracy_test)

# Calculate precision, recall, and f1 for the test set
precision_test = precision_score(test_df[0], test_pred, average='binary')  # binary classification
recall_test = recall_score(test_df[0], test_pred, average='binary')  # binary classification
f1_test = f1_score(test_df[0], test_pred, average='binary')  # binary classification

print("Test Precision:", precision_test)
print("Test Recall:", recall_test)
print("Test F1 Score:", f1_test)
