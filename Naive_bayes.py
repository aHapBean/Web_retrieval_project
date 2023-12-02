import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import re
import os
import nltk

model_filename = './model/naive_bayes/nb_moble.joblib'
vectorizer_filename = './model/naive_bayes/bayes_vectorizer.joblib'

# file_path = 'training.1600000.processed.noemoticon.csv'
# file_path = 'washed_data.csv'
# df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# X = df.iloc[:, 5].copy()
# y = df.iloc[:, 0].copy()
# print("start training...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Vectorize the text data using CountVectorizer
# vectorizer = CountVectorizer()
# X_train_vec = vectorizer.fit_transform(X_train)
# X_test_vec = vectorizer.transform(X_test)
#
# # Train the Multinomial Naive Bayes model
# nb_model = MultinomialNB()
# nb_model.fit(X_train_vec, y_train)
#
# # Make predictions on the test set
# y_pred = nb_model.predict(X_test_vec)
#
# # Print the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
#
# if not os.path.exists('./model/naive_bayes'):
#     os.makedirs('./model/naive_bayes')
# joblib.dump(nb_model, model_filename) # NOTE here
# joblib.dump(vectorizer, vectorizer_filename)

# NOTE 测试部分
loaded_model = joblib.load(model_filename)
loaded_vectorizer = joblib.load(vectorizer_filename)
test_df = pd.read_csv('testdata.manual.2009.06.14.csv', header=None, encoding='ISO-8859-1')
test_df[5] = test_df[5].apply(lambda x: re.sub('[^\w ]', '', str(x).lower()))
test_df[5] = test_df[5].apply(lambda x: nltk.tokenize.word_tokenize(str(x)))
test_df[5] = test_df[5].apply(lambda x: ' '.join(x))

test_data = test_df[5]  # 选择处理后的文本列
test_data_vec = loaded_vectorizer.transform(test_data)
test_pred = loaded_model.predict(test_data_vec)
accuracy = accuracy_score(test_df[0], test_pred)  # 使用原始标签列
# for i in range(len(test_df[0].tolist())):
#     if test_df[0][i] != test_pred[i]:
#         print("false class: ", test_pred[i], "err sentence: ", test_df[5][i])
# 0是悲伤
print(accuracy)
# nb 50000: 0.8133704735376045
# nb 100000: 81.05(76.15)
# nb 160万: 82.17(78.2)

